use std::f32::consts::SQRT_2;
use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

use simply_simd::{ Arch, Mask, Simd, enable_targets };

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{ Combiner, CombinerState };
use crate::noise::util::grid_data::SimplexGridData;
use crate::noise::util::grid_helpers::{
    Arena,
    ArenaBuffer,
    MaybeUninitSliceSimdExt,
    assume_init_slice,
    maybe_tail_load,
    maybe_tail_store,
    validate_grid_size,
    validate_state_size,
};
use crate::{ GridGenerator, Simplex };

const SQRT_3: f32 = 1.732_050_8;
const SKEW_2D: f32 = (SQRT_3 - 1.0) / 2.0;
const UNSKEW_2D: f32 = (3.0 - SQRT_3) / 6.0;

const SCALE: f32 = 80.0;
const SCALED_SQRT: f32 = (SQRT_2 / 2.0) * SCALE;

const A: f32 = SCALE;
const B: f32 = SCALED_SQRT;
const C: f32 = 0.0;
const X_GRADIENTS_2D: [f32; 8] = [A, B, C, -B, -A, -B, C, B];
const Y_GRADIENTS_2D: [f32; 8] = [C, B, A, B, C, -B, -A, -B];

const PRIME: u32 = 0x85ebca6b;
const BYTE_SHUFFLE: [u8; 64] = [
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13,
];

/// Resolves the gradient of a single lattice vertex `(i, j)`. Used only by the
/// scalar tail of a row. The SIMD path hashes whole chunks at once.
#[inline(always)]
fn gradient<A: Arch>(i: i32, j: i32, seed: u32) -> (f32, f32) {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(PRIME);
    let x_shuf = (
        Simd::<u32, A>::splat((i as u32).wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];
    let y_shuf = (
        Simd::<u32, A>::splat((j as u32).wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];
    let mix = x_shuf.wrapping_mul(y_shuf) ^ x_shuf;
    let idx = (mix >> 29) as usize;
    (X_GRADIENTS_2D[idx], Y_GRADIENTS_2D[idx])
}

/// Bounding lattice-vertex ranges `(i_min, i_max, j_min, j_max)` covered by the
/// whole output grid, sampled at the four extreme corners.
#[inline(always)]
fn lattice_bounds(grid_data: &SimplexGridData<2>) -> (i32, i32, i32, i32) {
    let wf = (grid_data.grid_size[0] as f32) - 1.0;
    let hf = (grid_data.grid_size[1] as f32) - 1.0;

    let corner = |px: f32, py: f32| {
        let xs = grid_data.origin[0] + grid_data.increment[0] * px;
        let ys = grid_data.origin[1] + grid_data.increment[1] * py;
        let s = (xs + ys) * SKEW_2D;
        ((xs + s).floor() as i32, (ys + s).floor() as i32)
    };

    let (i_tl, j_tl) = corner(0.0, 0.0);
    let (i_tr, j_tr) = corner(wf, 0.0);
    let (i_bl, j_bl) = corner(0.0, hf);
    let (i_br, j_br) = corner(wf, hf);

    let i_min = i_tl.min(i_tr).min(i_bl).min(i_br);
    let i_max = i_tl.max(i_tr).max(i_bl).max(i_br);
    let j_min = j_tl.min(j_tr).min(j_bl).min(j_br);
    let j_max = j_tl.max(j_tr).max(j_bl).max(j_br);
    (i_min, i_max, j_min, j_max)
}

/// Hashes every lattice vertex in `[i_min, i_max+1] x [j_min, j_max+1]` once
/// and stores the resolved gradient pairs into the two planar tables.
#[inline(always)]
fn build_gradient_table<A: Arch>(
    gx: &mut [MaybeUninit<f32>],
    gy: &mut [MaybeUninit<f32>],
    i_min: i32,
    j_min: i32,
    cols: usize,
    rows: usize,
    seed: u32
) {
    let lanes = Simd::<f32, A>::LANES;
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(PRIME);
    let seed_v = Simd::<u32, A>::splat(seed);
    let x_stride = Simd::<u32, A>::splat((lanes as u32).wrapping_mul(seed));

    for jv in 0..rows {
        let j = (j_min + (jv as i32)) as u32;
        let y_shuf = Simd::<u32, A>::splat(j.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime;

        let base = jv * cols;
        let mut x_vec =
            Simd::<u32, A>::splat((i_min as u32).wrapping_mul(seed)) +
            Simd::<u32, A>::iota(0) * seed_v;

        let mut col = 0;
        while col + lanes <= cols {
            let x_shuf = x_vec.permute_8(shuffle_indices) ^ prime;
            let indices = ((x_shuf * y_shuf) ^ x_shuf) >> 29;
            let gxv = indices.gather(&X_GRADIENTS_2D);
            let gyv = indices.gather(&Y_GRADIENTS_2D);
            unsafe {
                gx.write_simd(base + col, gxv);
                gy.write_simd(base + col, gyv);
            }
            x_vec += x_stride;
            col += lanes;
        }
        while col < cols {
            let (gxi, gyi) = gradient::<A>(i_min + (col as i32), j_min + (jv as i32), seed);
            unsafe {
                gx.get_unchecked_mut(base + col).write(gxi);
                gy.get_unchecked_mut(base + col).write(gyi);
            }
            col += 1;
        }
    }
}

/// Planar gradient tables, indexed by global lattice coordinates.
/// Initialised once per chunk and then reused, as it is only touched on the
/// chunk that enters a new lattice cell, never per sample.
struct GradientTable<'a> {
    gx: &'a [f32],
    gy: &'a [f32],
    i_min: i32,
    j_min: i32,
    cols: usize,
}

impl GradientTable<'_> {
    #[inline(always)]
    fn corner(&self, i: i32, j: i32) -> (f32, f32) {
        let idx = ((j - self.j_min) as usize) * self.cols + ((i - self.i_min) as usize);
        unsafe { (*self.gx.get_unchecked(idx), *self.gy.get_unchecked(idx)) }
    }
}

/// The three-corner simplex calculation. `x0`/`y0` are the
/// low-corner distances for this cell.
#[inline(always)]
fn simplex_calc(x0: f32, y0: f32, grads: &[(f32, f32); 4], upper: bool) -> f32 {
    let subbed_unskew = UNSKEW_2D - 1.0;
    let hi_skew_offset = 2.0 * UNSKEW_2D - 1.0;

    let (gx_lo, gy_lo) = grads[0];
    let (gx_mi, gy_mi) = if upper { grads[1] } else { grads[2] };
    let (gx_hi, gy_hi) = grads[3];

    let (x_mi, y_mi) = if upper {
        (x0 + subbed_unskew, y0 + UNSKEW_2D)
    } else {
        (x0 + UNSKEW_2D, y0 + subbed_unskew)
    };
    let (x_hi, y_hi) = (x0 + hi_skew_offset, y0 + hi_skew_offset);

    let t_lo_pre = 0.5 - x0.mul_add(x0, y0 * y0);
    let t_mi = (0.5 - x_mi.mul_add(x_mi, y_mi * y_mi)).max(0.0);
    let t_hi_pre = t_lo_pre + ((2.0 * SQRT_3) / 3.0).mul_add(x0 + y0, -2.0 / 3.0);
    let t_lo = t_lo_pre.max(0.0);
    let t_hi = t_hi_pre.max(0.0);

    let t2_lo = t_lo * t_lo;
    let t2_mi = t_mi * t_mi;
    let t2_hi = t_hi * t_hi;

    let dot_lo = gx_lo.mul_add(x0, gy_lo * y0);
    let dot_mi = gx_mi.mul_add(x_mi, gy_mi * y_mi);
    let dot_hi = gx_hi.mul_add(x_hi, gy_hi * y_hi);

    let t4_lo = t2_lo * t2_lo;
    let t4_mi = t2_mi * t2_mi;
    let t4_hi = t2_hi * t2_hi;
    t4_lo.mul_add(dot_lo, t4_mi.mul_add(dot_mi, t4_hi * dot_hi))
}

/// Vectorised simplex calculation for `LANES` samples that may each belong to their
/// own cell.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn simplex_calc_simd<A: Arch>(
    x0: Simd<f32, A>,
    y0: Simd<f32, A>,
    t_lo_pre: Simd<f32, A>,
    gx_lo: Simd<f32, A>,
    gy_lo: Simd<f32, A>,
    gx_mi: Simd<f32, A>,
    gy_mi: Simd<f32, A>,
    gx_hi: Simd<f32, A>,
    gy_hi: Simd<f32, A>,
    upper: Mask<f32, A>
) -> Simd<f32, A> {
    let subbed_unskew = Simd::<f32, A>::splat(UNSKEW_2D - 1.0);
    let hi_skew_offset = Simd::<f32, A>::splat(2.0 * UNSKEW_2D - 1.0);
    let unskew = Simd::<f32, A>::splat(UNSKEW_2D);
    let half = Simd::<f32, A>::splat(0.5);
    let zero = Simd::<f32, A>::splat(0.0);
    let t_hi_coef = Simd::<f32, A>::splat((2.0 * SQRT_3) / 3.0);
    let neg_two_thirds = Simd::<f32, A>::splat(-2.0 / 3.0);

    let x_mi = x0 + upper.select(subbed_unskew, unskew);
    let y_mi = y0 + upper.select(unskew, subbed_unskew);
    let (x_hi, y_hi) = (x0 + hi_skew_offset, y0 + hi_skew_offset);

    let t_mi = (half - x_mi.mul_add(x_mi, y_mi * y_mi)).max(zero);
    let t_hi_pre = t_lo_pre + t_hi_coef.mul_add(x0 + y0, neg_two_thirds);
    let t_lo = t_lo_pre.max(zero);
    let t_hi = t_hi_pre.max(zero);

    let t2_lo = t_lo * t_lo;
    let t2_mi = t_mi * t_mi;
    let t2_hi = t_hi * t_hi;

    let dot_lo = gx_lo.mul_add(x0, gy_lo * y0);
    let dot_mi = gx_mi.mul_add(x_mi, gy_mi * y_mi);
    let dot_hi = gx_hi.mul_add(x_hi, gy_hi * y_hi);

    let t4_lo = t2_lo * t2_lo;
    let t4_mi = t2_mi * t2_mi;
    let t4_hi = t2_hi * t2_hi;
    t4_lo.mul_add(dot_lo, t4_mi.mul_add(dot_mi, t4_hi * dot_hi))
}

/// Debug-only tally of which SIMD path each chunk took, plus sampled per-stage
/// cycle counts. Populated when `QN_SPLIT_STATS` is set.
#[derive(Default, Clone, Copy)]
struct RowSplitCounts {
    uniform: u64,
    fallback: u64,
    tail: u64,
    /// Whether per-stage sampling is active (set once from `QN_SPLIT_STATS`).
    timing: bool,
    /// Sample one in every `sample_every` chunk iterations.
    sample_every: u32,
    /// Countdown until the next sampled iteration.
    sample_counter: u32,
    /// Number of chunk iterations sampled so far.
    samples: u64,
    /// A sampled chunk iteration's total RDTSC cycles: floors, extracts,
    /// branch, value computation, fill, and advances.
    iter_cycles: u64,
    /// Sampled cycle cost of the uniform path's value computation (table
    /// lookups, splats, kernel).
    compute_uniform_cycles: u64,
    /// Sampled cycle cost of the fallback path's value computation (hashing,
    /// gathers, kernel).
    compute_fallback_cycles: u64,
    /// Sampled cycle cost of the combiner `fill_block` call.
    fill_cycles: u64,
    /// Wall-clock time of the whole rows loop, for calibration.
    rows_wall: Duration,
}

impl RowSplitCounts {
    #[inline(always)]
    fn now() -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe { std::arch::x86_64::_rdtsc() }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            0
        }
    }

    /// Marks the start of a chunk iteration, sampling one in `sample_every`
    /// when timing is enabled. Returns a non-zero clock value for a sampled
    /// iteration, or 0 to skip.
    #[inline(always)]
    fn tick(&mut self) -> u64 {
        if !self.timing {
            return 0;
        }
        if self.sample_counter > 0 {
            self.sample_counter -= 1;
            return 0;
        }
        self.sample_counter = self.sample_every;
        let t = Self::now();
        if t != 0 {
            self.samples += 1;
        }
        t
    }

    /// Closes the value-computation span of a sampled iteration, routing the
    /// cycles into the uniform or fallback.
    #[inline(always)]
    fn compute_done(&mut self, t0: u64, uniform: bool) -> u64 {
        if t0 == 0 {
            return 0;
        }
        let t1 = Self::now();
        let cycles = t1.wrapping_sub(t0);
        if uniform {
            self.compute_uniform_cycles += cycles;
        } else {
            self.compute_fallback_cycles += cycles;
        }
        t1
    }

    /// Closes the `fill_block` span of a sampled iteration.
    #[inline(always)]
    fn fill_done(&mut self, t1: u64) {
        if t1 == 0 {
            return;
        }
        self.fill_cycles += Self::now().wrapping_sub(t1);
    }

    /// Closes the whole chunk iteration (including the vector advances),
    /// producing the iteration total from which `compute` and `fill` are
    /// subtracted to get the remaining loop overhead.
    #[inline(always)]
    fn iter_done(&mut self, t0: u64) {
        if t0 == 0 {
            return;
        }
        self.iter_cycles += Self::now().wrapping_sub(t0);
    }
}

#[enable_targets(A)]
impl GridGenerator<2> for Simplex {
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<2>,
        combiner: C::Config,
        state: &mut [f32],
        dst: &mut [f32]
    ) {
        validate_grid_size(params.grid_size, dst.len());
        validate_state_size::<C, A, _>(params.grid_size, dst.len());

        let grid_data = SimplexGridData::new(&params);
        let row_width = grid_data.grid_size[0];

        let (i_min, i_max, j_min, j_max) = lattice_bounds(&grid_data);
        let cols = (i_max - i_min + 2) as usize;
        let rows = (j_max - j_min + 2) as usize;
        let mut cache = ArenaBuffer::<A>::with_capacity(cols * rows * 2);
        let mut arena = Arena::with_cache(&mut cache);
        let gx = arena.allocate::<f32>(cols * rows);
        let gy = arena.allocate::<f32>(cols * rows);
        build_gradient_table::<A>(gx, gy, i_min, j_min, cols, rows, params.seed);
        let table = GradientTable {
            gx: unsafe {
                assume_init_slice(gx)
            },
            gy: unsafe {
                assume_init_slice(gy)
            },
            i_min,
            j_min,
            cols,
        };

        let mut counters = RowSplitCounts::default();
        counters.timing = std::env::var_os("QN_SPLIT_STATS").is_some();
        counters.sample_every = 16;

        let rows_clock = if counters.timing {
            Some(Instant::now())
        } else {
            None
        };

        for oy in 0..grid_data.grid_size[1] {
            simplex_fill_row::<A, C, INIT, FINAL>(
                &grid_data,
                &table,
                &combiner,
                params.seed,
                oy,
                row_width,
                &mut counters,
                state,
                dst
            );
        }

        if let Some(started) = rows_clock {
            counters.rows_wall = started.elapsed();
        }

        if counters.timing {
            let iter = counters.iter_cycles.max(1) as f64;
            let measured = (counters.compute_uniform_cycles
                + counters.compute_fallback_cycles
                + counters.fill_cycles) as f64;
            eprintln!(
                "simplex split: uniform={} fallback={} tail={} full_simd={} ({} rows x {} cols, {} lanes)",
                counters.uniform,
                counters.fallback,
                counters.tail,
                counters.uniform + counters.fallback,
                grid_data.grid_size[1],
                grid_data.grid_size[0],
                Simd::<f32, A>::LANES
            );
            eprintln!(
                "simplex timing: fill={:.1}us samples={} avg_iter={:.0}cy | compute(uniform)={:.0}% compute(fallback)={:.0}% fill={:.0}% other={:.0}%",
                counters.rows_wall.as_secs_f64() * 1e6,
                counters.samples,
                iter / counters.samples.max(1) as f64,
                100.0 * counters.compute_uniform_cycles as f64 / iter,
                100.0 * counters.compute_fallback_cycles as f64 / iter,
                100.0 * counters.fill_cycles as f64 / iter,
                100.0 * (1.0 - measured / iter),
            );
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn simplex_fill_row<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
    grid_data: &SimplexGridData<2>,
    table: &GradientTable,
    combiner: &C::Config,
    seed: u32,
    oy: usize,
    row_width: usize,
    counters: &mut RowSplitCounts,
    state: &mut [f32],
    dst: &mut [f32]
) {
    let lanes = Simd::<f32, A>::LANES;
    let lanes_f = lanes as f32;
    let weight = grid_data.weight;
    let row_start = oy * row_width;

    // Per-step skewed-space increments along the x-axis.
    let dsx = grid_data.increment[0] * (1.0 + SKEW_2D);
    let dsy = grid_data.increment[0] * SKEW_2D;

    // Sample-space y is fixed across the whole row.
    let y = grid_data.origin[1] + (oy as f32) * grid_data.increment[1];

    let s0 = (grid_data.origin[0] + y) * SKEW_2D;
    let sx = grid_data.origin[0] + s0;
    let sy = y + s0;

    // Lane-index vector and per-step skewed increments.
    let iota_f = Simd::<i32, A>::iota(0).cast_float();
    let sx_stride = Simd::<f32, A>::splat(dsx * lanes_f);
    let sy_stride = Simd::<f32, A>::splat(dsy * lanes_f);

    // Running vectors of sample-space x and skewed coords for the current
    // chunk, advanced by `lanes` each iteration.
    let x0_stride = Simd::<f32, A>::splat(grid_data.increment[0] * lanes_f);
    let mut lane_x0 =
        Simd::<f32, A>::splat(grid_data.origin[0]) +
        iota_f * Simd::<f32, A>::splat(grid_data.increment[0]);
    let mut sxv = Simd::<f32, A>::splat(sx) + iota_f * Simd::<f32, A>::splat(dsx);
    let mut syv = Simd::<f32, A>::splat(sy) + iota_f * Simd::<f32, A>::splat(dsy);

    let unskew_v = Simd::<f32, A>::splat(UNSKEW_2D);
    let y_v = Simd::<f32, A>::splat(y);
    let weight_v = Simd::<f32, A>::splat(weight);
    let half_v = Simd::<f32, A>::splat(0.5);

    // Hash constants.
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(PRIME);
    let seed_v = Simd::<u32, A>::splat(seed);

    // Lane-rotate for the chunk-uniformity test: result[i] = v[i+1].
    let shift_by_one = Simd::<u32, A>::from_slice(&[1, 2, 3, 4, 5, 6, 7, 0]);

    let mut ox = 0;

    while ox + lanes <= row_width {
        let t0 = counters.tick();

        let ic = sxv.floor();
        let jc = syv.floor();
        let ic_v = ic.cast_int_trunc();
        let jc_v = jc.cast_int_trunc();

        let eq = ic_v.simd_eq(ic_v.permute_32(shift_by_one))
            & jc_v.simd_eq(jc_v.permute_32(shift_by_one));
        let uniform = (!eq).all_false();

        let value = if uniform {
            counters.uniform += 1;
            // Whole chunk shares one lattice cell. Get its four corners
            // from the table and splat the gradients.
            let ic_arr = ic_v.to_array();
            let jc_arr = jc_v.to_array();
            let sum = (ic + jc) * unskew_v;
            let cx = ic - sum;
            let cy = jc - sum;
            let x0 = lane_x0 - cx;
            let y0 = y_v - cy;
            let upper = x0.simd_gt(y0);

            // Get the gradient values for each corner.
            let (lo_x, lo_y) = table.corner(ic_arr[0], jc_arr[0]);
            let (up_x, up_y) = table.corner(ic_arr[0] + 1, jc_arr[0]);
            let (lw_x, lw_y) = table.corner(ic_arr[0], jc_arr[0] + 1);
            let (hi_x, hi_y) = table.corner(ic_arr[0] + 1, jc_arr[0] + 1);

            // Mid corner
            let gx_mi = upper.select(Simd::<f32, A>::splat(up_x), Simd::<f32, A>::splat(lw_x));
            let gy_mi = upper.select(Simd::<f32, A>::splat(up_y), Simd::<f32, A>::splat(lw_y));

            let t_lo_pre = half_v - x0.mul_add(x0, y0 * y0);
            simplex_calc_simd::<A>(
                x0,
                y0,
                t_lo_pre,
                Simd::<f32, A>::splat(lo_x),
                Simd::<f32, A>::splat(lo_y),
                gx_mi,
                gy_mi,
                Simd::<f32, A>::splat(hi_x),
                Simd::<f32, A>::splat(hi_y),
                upper
            ) * weight_v
        } else {
            counters.fallback += 1;
            // Chunk spans cell boundaries: hash every lane's corners in SIMD,
            // skipping the table
            let sum = (ic + jc) * unskew_v;
            let cx = ic - sum;
            let cy = jc - sum;
            let x0 = lane_x0 - cx;
            let y0 = y_v - cy;
            let upper = x0.simd_gt(y0);

            // Hash the low and high lattice corners of each lane's cell.
            let ic_u = ic_v.raw_cast::<u32>();
            let jc_u = jc_v.raw_cast::<u32>();
            let x1 = ic_u * seed_v;
            let y1 = jc_u * seed_v;
            let x2 = x1 + seed_v;
            let y2 = y1 + seed_v;

            let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
            let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
            let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
            let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;

            let mix_lo = (x1_shuf * y1_shuf) ^ x1_shuf;
            let mix_hi = (x2_shuf * y2_shuf) ^ x2_shuf;

            // Mid corner
            let upper_u = upper.raw_cast::<u32>();
            let x_shuf_mi = upper_u.select(x2_shuf, x1_shuf);
            let y_shuf_mi = upper_u.select(y1_shuf, y2_shuf);
            let mix_mi = (x_shuf_mi * y_shuf_mi) ^ x_shuf_mi;

            let indices_lo = mix_lo >> 29;
            let indices_mi = mix_mi >> 29;
            let indices_hi = mix_hi >> 29;

            let t_lo_pre = half_v - x0.mul_add(x0, y0 * y0);

            let gx_lo = indices_lo.gather(&X_GRADIENTS_2D);
            let gy_lo = indices_lo.gather(&Y_GRADIENTS_2D);
            let gx_mi = indices_mi.gather(&X_GRADIENTS_2D);
            let gy_mi = indices_mi.gather(&Y_GRADIENTS_2D);
            let gx_hi = indices_hi.gather(&X_GRADIENTS_2D);
            let gy_hi = indices_hi.gather(&Y_GRADIENTS_2D);
            simplex_calc_simd::<A>(
                x0,
                y0,
                t_lo_pre,
                gx_lo,
                gy_lo,
                gx_mi,
                gy_mi,
                gx_hi,
                gy_hi,
                upper
            ) * weight_v
        };

        let t1 = counters.compute_done(t0, uniform);

        fill_block::<A, C, INIT, FINAL, false>(
            grid_data,
            combiner,
            state,
            dst,
            row_start + ox,
            lanes,
            value
        );
        counters.fill_done(t1);

        // Advance all tracking by LANES steps.
        sxv += sx_stride;
        syv += sy_stride;
        lane_x0 += x0_stride;
        ox += lanes;
        counters.iter_done(t0);
    }

    // Scalar tail: remaining samples after the last full SIMD chunk
    let mut sx = grid_data.origin[0] + s0 + dsx * (ox as f32);
    let mut sy = y + s0 + dsy * (ox as f32);
    while ox < row_width {
        counters.tail += 1;
        let ic = sx.floor() as i32;
        let jc = sy.floor() as i32;
        let c = grid_data.unskew(&[ic, jc]);
        let x = grid_data.origin[0] + (ox as f32) * grid_data.increment[0];
        let x0 = x - c[0];
        let y0 = y - c[1];
        let upper = x0 > y0;

        let grads = [
            gradient::<A>(ic, jc, seed),
            gradient::<A>(ic + 1, jc, seed),
            gradient::<A>(ic, jc + 1, seed),
            gradient::<A>(ic + 1, jc + 1, seed),
        ];
        let value = Simd::<f32, A>::splat(simplex_calc(x0, y0, &grads, upper) * weight);

        fill_block::<A, C, INIT, FINAL, true>(
            grid_data,
            combiner,
            state,
            dst,
            row_start + ox,
            1,
            value
        );

        sx += dsx;
        sy += dsy;
        ox += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fill_block<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool, const IS_TAIL: bool>(
    grid_data: &SimplexGridData<2>,
    combiner: &C::Config,
    state: &mut [f32],
    dst: &mut [f32],
    sample_start: usize,
    lanes: usize,
    value: Simd<f32, A>
) {
    let sample_end = sample_start + lanes;

    let (cur_state, mut result) = if INIT {
        C::initialize_sample(combiner, value)
    } else {
        let mut cur_state = C::State::<A>::default();
        for i in 0..C::State::<A>::STATE_SIZE {
            let offset = i * grid_data.total_size;
            cur_state[i] = unsafe {
                maybe_tail_load::<A, IS_TAIL>(sample_start + offset..sample_end + offset, state)
            };
        }
        let cur_result = unsafe { maybe_tail_load::<A, IS_TAIL>(sample_start..sample_end, dst) };
        C::apply_sample(combiner, cur_state, cur_result, value)
    };

    if !FINAL {
        for i in 0..C::State::<A>::STATE_SIZE {
            let offset = i * grid_data.total_size;
            unsafe {
                maybe_tail_store::<A, IS_TAIL>(
                    sample_start + offset..sample_end + offset,
                    cur_state[i],
                    state
                );
            }
        }
    }

    if FINAL {
        result = C::finalize_sample(combiner, cur_state, result);
    }

    unsafe { maybe_tail_store::<A, IS_TAIL>(sample_start..sample_end, result, dst) }
}

#[cfg(test)]
mod tests {
    use simply_simd::Simd;

    use crate::api::batch::interface::BatchGenerator;
    use crate::api::seed::gen_octave_seed;
    use crate::math::random::Random;
    use crate::simd::StaticArch;
    use crate::{ Fbm, Grid, Simplex };
    #[cfg(feature = "image")]
    use crate::emit::NoiseImageExt;

    fn reference(seed: u32, px: f32, py: f32, freq: f32) -> f32 {
        let gain = Simplex::sample_batch::<StaticArch>(
            seed,
            [Simd::splat(px), Simd::splat(py)],
            [Simd::splat(freq), Simd::splat(freq)]
        );
        gain.to_array()[0]
    }

    #[test]
    fn simplex_grid_2d_reference() {
        let seed = 123456789i64;

        for freq in [1.0 / 32.0, 1.0 / 8.0, 1.0 / 6.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0] {
            check_reference(64, 64, seed, -5, 3, freq);
        }

        check_reference(32, 96, seed, -5, 3, 1.0 / 6.0);
    }

    fn check_reference(w: usize, h: usize, seed: i64, offset_x: i32, offset_y: i32, freq: f32) {
        let grid = Grid::<2>::new(w, h).seed(seed).sample_position(offset_x, offset_y);
        let grid_seed = Random::mix_u64(seed as u64);
        let base_seed = Random::mix_u64_pair(grid_seed, 0xd5e7b3c94f8a1e6b);
        let octave_seed = gen_octave_seed([freq, freq], base_seed);

        let mut result = vec![0.0; w * h];
        grid.builder::<Fbm, Simplex>().frequency(freq).fill(result.as_mut_slice());

        let mut max_diff = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let px = (offset_x + (x as i32)) as f32;
                let py = (offset_y + (y as i32)) as f32;
                let reference = reference(octave_seed, px, py, freq);
                let actual = result[y * w + x];
                max_diff = max_diff.max((actual - reference).abs());
            }
        }
        assert!(
            max_diff < 1e-4,
            "Grid simplex at freq {freq} diverges from the brute-force Simplex by {max_diff}"
        );
    }

    #[test]
    #[cfg(feature = "image")]
    fn grid_image() {
        let grid = Grid::<2>::new(256, 256).seed(42).sample_position(-128, -128);

        grid.builder::<Fbm, Simplex>()
            .frequency(1.0 / 32.0)
            .into_iter()
            .to_grayscale_image(256, 256, "test_images/grid_2d_simplex_seeded.png");
    }
}
