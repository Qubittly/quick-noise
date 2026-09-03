use std::f32::consts::SQRT_2;

use simply_simd::{ Arch, Simd, enable_targets };

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{ Combiner, CombinerState };
use crate::noise::util::grid_data::SimplexGridData;
use crate::noise::util::grid_helpers::{
    maybe_tail_load,
    maybe_tail_store,
    validate_grid_size,
    validate_state_size,
};
use crate::{ GridGenerator, Simplex };

const SQRT_3: f32 = 1.732_050_8;
const UNSKEW_2D: f32 = (3.0 - SQRT_3) / 6.0;

const SCALE: f32 = 80.0;
const SCALED_SQRT: f32 = (SQRT_2 / 2.0) * SCALE;

const A: f32 = SCALE;
const B: f32 = SCALED_SQRT;
const C: f32 = 0.0;
const X_GRADIENTS_2D: [f32; 8] = [A, B, C, -B, -A, -B, C, B];
const Y_GRADIENTS_2D: [f32; 8] = [C, B, A, B, C, -B, -A, -B];

const BYTE_SHUFFLE: [u8; 64] = [
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13,
];

#[inline(always)]
fn hash_cell<A: Arch>(x: u32, y: u32, seed: u32) -> u32 {
    let shuffle_indices: Simd<u8, A> = unsafe {
        Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..])
    };
    let prime = Simd::<u32, A>::splat(0x85ebca6b_u32);
    let x_shuf = (
        Simd::<u32, A>::splat(x.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];
    let y_shuf = (
        Simd::<u32, A>::splat(y.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];

    x_shuf.wrapping_mul(y_shuf) ^ x_shuf
}

/// Resolves the four corner gradients of lattice cell `(i, j)`:
/// `[ (i,j), (i+1,j), (i,j+1), (i+1,j+1) ]`. All four are computed here once per cell.
#[inline(always)]
fn cell_gradients<A: Arch>(i: i32, j: i32, seed: u32) -> [(f32, f32); 4] {
    let corner = |ii: i32, jj: i32| {
        let idx = (hash_cell::<A>(ii as u32, jj as u32, seed) >> 29) as usize;
        (X_GRADIENTS_2D[idx], Y_GRADIENTS_2D[idx])
    };

    [corner(i, j), corner(i + 1, j), corner(i, j + 1), corner(i + 1, j + 1)]
}

/// The three-corner simplex sum. `x0`/`y0` are the
/// low-corner distances for this cell..
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

        let [i_lo, i_hi, j_lo, j_hi] = grid_data.cell_bounds();

        // iterate over the skewed lattice cells that intersect the output grid.
        for j in j_lo..=j_hi {
            for i in i_lo..=i_hi {
                // Clip the cell's rhombus/diamond to a small output sample range.
                let Some([ox_lo, ox_hi, oy_lo, oy_hi]) = grid_data.cell_bbox(i, j) else {
                    continue;
                };

                // corner gradients stay constant across the whole cell
                let grads = cell_gradients::<A>(i, j, params.seed);

                // Unskewed low corner `(cx, cy)` of this cell: the sample's
                // low-corner distances are `(x_lo - cx, y_lo - cy)`.
                let [cx, cy] = grid_data.unskew(i, j);

                // iterate over the output samples 
                for oy in oy_lo..oy_hi {
                    let y_lo = grid_data.origin[1] + oy as f32 * grid_data.increment[1];
                    let y0 = y_lo - cy;

                    for ox in ox_lo..ox_hi {
                        let x_lo = grid_data.origin[0] + ox as f32 * grid_data.increment[0];
                        let [sx, sy] = grid_data.skew(x_lo, y_lo);
                        let x0 = x_lo - cx;

                        // This output sample is only produced by the
                        // cell whose skewed floor matches its skewed coordinate.
                        if sx.floor() as i32 == i && sy.floor() as i32 == j {
                            let upper = x0 > y0;

                            let value = simplex_calc(x0, y0, &grads, upper) * grid_data.weight;

                            let sample_start = (oy as usize) * row_width + (ox as usize);
                            let sample_end = sample_start + 1;

                            let (cur_state, mut result) = if INIT {
                                C::initialize_sample(&combiner, Simd::splat(value))
                            } else {
                                let mut cur_state = C::State::<A>::default();
                                for k in 0..C::State::<A>::STATE_SIZE {
                                    let offset = k * grid_data.total_size;
                                    cur_state[k] = unsafe {
                                        maybe_tail_load::<A, true>(
                                            sample_start + offset..sample_end + offset,
                                            state,
                                        )
                                    };
                                }
                                let cur_result = unsafe {
                                    maybe_tail_load::<A, true>(sample_start..sample_end, dst)
                                };
                                C::apply_sample(
                                    &combiner,
                                    cur_state,
                                    cur_result,
                                    Simd::splat(value),
                                )
                            };

                            if !FINAL {
                                for k in 0..C::State::<A>::STATE_SIZE {
                                    let offset = k * grid_data.total_size;
                                    unsafe {
                                        maybe_tail_store::<A, true>(
                                            sample_start + offset..sample_end + offset,
                                            cur_state[k],
                                            state,
                                        );
                                    }
                                }
                            }

                            if FINAL {
                                result = C::finalize_sample(&combiner, cur_state, result);
                            }

                            unsafe {
                                maybe_tail_store::<A, true>(sample_start..sample_end, result, dst);
                            }
                        }
                    }
                }
            }
        }
    }
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
        // Cellular Grid Test
        let grid = Grid::<2>::new(256, 256).seed(42).sample_position(-128, -128);

        grid.builder::<Fbm, Simplex>()
            .frequency(1.0 / 32.0)
            .into_iter()
            .to_grayscale_image(256, 256, "test_images/grid_2d_simplex_seeded.png");
    }
}
