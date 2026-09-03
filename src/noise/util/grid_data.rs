use std::array::from_fn;
use std::mem::MaybeUninit;

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::util::grid_helpers::{Arena, configure_tiling, fill_grid_indices};
use crate::simd::Arch;
use crate::simd::register::Simd;

pub(crate) struct GridData<'a, const D: usize> {
    pub total_size: usize,
    pub weight: f32,
    pub grid_size: [usize; D],
    pub grid_start: [i32; D],
    pub num_loops: [usize; D],
    pub octave_tiling: [Option<u32>; D],
    pub distances: [&'a mut [MaybeUninit<f32>]; D],
    pub grid_indices: [&'a mut [MaybeUninit<u32>]; D],
}

impl<'a, const D: usize> GridData<'a, D> {
    #[inline(always)]
    pub fn new<A: Arch>(
        params: &GridNoiseParams<D>,
        arena: &mut Arena<'a>,
        padded_size: &[usize; D],
    ) -> Self {
        let lanes = Simd::<f32, A>::LANES;

        let total_size = params.grid_size.iter().product();
        let increment: [f32; D] = from_fn(|i| params.frequency[i] * params.magnification);

        // Get the starting gradient coordinates and how far the first sample is to the next one
        let grid_start: [i32; D] =
            from_fn(|i| (params.position[i] as f32 * increment[i]).floor() as i32);

        let frac_start: [f32; D] =
            from_fn(|i| (params.position[i] as f32 * increment[i] - grid_start[i] as f32).max(0.0));

        // Only the raw fractional distance is needed to locate cell boundaries
        let distances = from_fn(|i| arena.allocate(padded_size[i]));

        let mut cur_dist: [_; D] = from_fn(|i| {
            Simd::<f32, A>::iota(0.0) * Simd::<f32, A>::splat(increment[i])
                + Simd::<f32, A>::splat(frac_start[i])
        });
        let chunk_increment: [_; D] =
            from_fn(|i| Simd::<f32, A>::splat(increment[i] * lanes as f32));

        for axis in 0..D {
            for i in (0..params.grid_size[axis]).step_by(lanes) {
                let fract_dist = cur_dist[axis].fract();
                unsafe {
                    fract_dist.copy_to_aligned_slice_unchecked(
                        distances[axis].get_unchecked_mut(i..).assume_init_mut(),
                    );
                }
                cur_dist[axis] += chunk_increment[axis];
            }
        }

        // Identify the cutoff points between frequency-based grid boundaries.
        let mut grid_indices = from_fn(|i| arena.allocate(padded_size[i]));
        let num_loops = fill_grid_indices::<A, D>(&mut grid_indices, &distances, params.grid_size);

        // Adjust the tiling.
        let octave_tiling = configure_tiling(params);

        Self {
            total_size,
            weight: params.weight,
            grid_size: params.grid_size,
            grid_start,
            num_loops,
            octave_tiling,
            distances,
            grid_indices,
        }
    }
}


pub(crate) struct GridDataLerp<'a, const D: usize> {
    pub total_size: usize,
    pub weight: f32,
    pub grid_size: [usize; D],
    pub grid_start: [i32; D],
    pub increment: [f32; D],
    pub num_loops: [usize; D],
    pub octave_tiling: [Option<u32>; D],
    pub distances: [&'a mut [MaybeUninit<f32>]; D],
    pub fade_factors: [&'a mut [MaybeUninit<f32>]; D],
    pub grid_indices: [&'a mut [MaybeUninit<u32>]; D],
}

#[repr(u8)]
pub(crate) enum Lerp {
    Cubic = 0,
    Quintic = 1,
}


impl Lerp {
    #[inline(always)]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Cubic,
            1 => Self::Quintic,
            _ => unreachable!(),
        }
    }
}

impl<'a, const D: usize> GridDataLerp<'a, D> {
    #[inline(always)]
    pub fn new<A: Arch, const LERP: u8>(
        params: &GridNoiseParams<D>,
        arena: &mut Arena<'a>,
        padded_size: &[usize; D],
    ) -> Self {
        let lerp_type = Lerp::from_u8(LERP);
        let lanes = Simd::<f32, A>::LANES;

        let total_size = params.grid_size.iter().product();
        let increment = from_fn(|i| params.frequency[i] * params.magnification);

        // Get the starting gradient coordinates and how far the first sample is to the next one.
        let grid_start: [i32; D] =
            from_fn(|i| (params.position[i] as f32 * increment[i]).floor() as i32);

        let frac_start: [f32; D] =
            from_fn(|i| (params.position[i] as f32 * increment[i] - grid_start[i] as f32).max(0.0));

        // Quintic lerp the distances to get the fade factor.
        let distances = from_fn(|i| arena.allocate(padded_size[i]));
        let fade_factors = from_fn(|i| arena.allocate(padded_size[i]));

        // Get the distances from the gradient gridpoints.
        let mut cur_dist: [_; D] = from_fn(|i| {
            Simd::<f32, A>::iota(0.0) * Simd::<f32, A>::splat(increment[i])
                + Simd::<f32, A>::splat(frac_start[i])
        });
        let chunk_increment: [_; D] =
            from_fn(|i| Simd::<f32, A>::splat(increment[i] * lanes as f32));

        for axis in 0..D {
            for i in (0..params.grid_size[axis]).step_by(lanes) {
                let fract_dist = cur_dist[axis].fract();
                let cur_lerp = match lerp_type {
                    Lerp::Cubic => fract_dist.cubic_lerp(),
                    Lerp::Quintic => fract_dist.quintic_lerp(),
                };

                unsafe {
                    fract_dist.copy_to_aligned_slice_unchecked(
                        distances[axis].get_unchecked_mut(i..).assume_init_mut(),
                    );
                    cur_lerp.copy_to_aligned_slice_unchecked(
                        fade_factors[axis].get_unchecked_mut(i..).assume_init_mut(),
                    );
                }
                cur_dist[axis] += chunk_increment[axis];
            }
        }

        // Identify the cutoff points between frequency-based grid boundaries .
        let mut grid_indices = from_fn(|i| arena.allocate(padded_size[i]));
        let num_loops = fill_grid_indices::<A, D>(&mut grid_indices, &distances, params.grid_size);

        // Adjust the tiling.
        let octave_tiling = configure_tiling(params);

        Self {
            total_size,
            weight: params.weight,
            grid_size: params.grid_size,
            grid_start,
            increment,
            num_loops,
            octave_tiling,
            distances,
            fade_factors,
            grid_indices,
        }
    }
}

const SQRT_3: f32 = 1.732_050_8;
const SKEW_2D: f32 = (SQRT_3 - 1.0) / 2.0;
const UNSKEW_2D: f32 = (3.0 - SQRT_3) / 6.0;

/// Simplex's lattice is skew-transformed, so cells are diagonal and are
/// per_celld directly. Per-cell we resolve the four corner gradients once and
/// use them over the output samples owned by that cell.
pub(crate) struct SimplexGridData<'a> {
    pub total_size: usize,
    pub weight: f32,
    pub grid_size: [usize; 2],
    pub increment: [f32; 2],
    /// Position of the first sample in output space (`position * increment`).
    pub origin: [f32; 2],
    /// Skewed lattice index `(i0, j0)` that the first sample (`origin`) falls into.
    pub grid_start: [i32; 2],
    pub octave_tiling: [Option<u32>; 2],
    /// Whether a cell covers more than ~1 sample, i.e. whether rasterizing
    /// cells is worthwhile or we just do a plain per-sample fallback. Not used right now 
    pub per_cell: bool,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> SimplexGridData<'a> {
    #[inline(always)]
    pub fn new(
        params: &GridNoiseParams<2>,
    ) -> Self {
        let total_size = params.grid_size.iter().product();
        let increment = [params.frequency[0] * params.magnification, params.frequency[1] * params.magnification];
        let origin = [params.position[0] as f32 * increment[0], params.position[1] as f32 * increment[1]];

        // Skew the region's first sample to locate the enclosing lattice cell.
        let s = (origin[0] + origin[1]) * SKEW_2D;
        let grid_start = [(origin[0] + s).floor() as i32, (origin[1] + s).floor() as i32];

        let octave_tiling = configure_tiling(params);

        Self {
            total_size,
            weight: params.weight,
            grid_size: params.grid_size,
            increment,
            origin,
            grid_start,
            octave_tiling,
            per_cell: increment[0] < 1.0 && increment[1] < 1.0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Skew a sample-space coordinate into the skewed lattice coordinate space `(i, j)`.
    #[inline(always)]
    pub fn skew(&self, x: f32, y: f32) -> [f32; 2] {
        let s = (x + y) * SKEW_2D;
        [x + s, y + s]
    }

    /// True sample-space position `(x, y)` of a lattice corner `(i, j)`.
    #[inline(always)]
    pub fn unskew(&self, i: i32, j: i32) -> [f32; 2] {
        let t = (i as f32 + j as f32) * UNSKEW_2D;
        [i as f32 - t, j as f32 - t]
    }

    /// `(i_lo, i_hi, j_lo, j_hi)` range of lattice cells that can
    /// intersect the output grid, padded by one cell for safety.
    #[inline(always)]
    pub fn cell_bounds(&self) -> [i32; 4] {
        let corners = [
            (0, 0),
            (self.grid_size[0] as i32, 0),
            (0, self.grid_size[1] as i32),
            (self.grid_size[0] as i32, self.grid_size[1] as i32),
        ];

        let (mut i_lo, mut i_hi, mut j_lo, mut j_hi) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
        for (ox, oy) in corners {
            let x = self.origin[0] + ox as f32 * self.increment[0];
            let y = self.origin[1] + oy as f32 * self.increment[1];
            let [sx, sy] = self.skew(x, y);
            let (i, j) = (sx.floor() as i32, sy.floor() as i32);
            i_lo = i_lo.min(i);
            i_hi = i_hi.max(i);
            j_lo = j_lo.min(j);
            j_hi = j_hi.max(j);
        }

        [i_lo - 1, i_hi + 1, j_lo - 1, j_hi + 1]
    }

    /// Axis-aligned bounding box of a cell's diamond, clipped to the grid, as
    /// output-sample index ranges `(ox_lo, ox_hi, oy_lo, oy_hi)`. This is the
    /// *rasterization range* only: ownership of a sample is decided separately.
    #[inline(always)]
    pub fn cell_bbox(&self, i: i32, j: i32) -> Option<[i32; 4]> {
        let corners = [
            self.unskew(i, j),
            self.unskew(i + 1, j),
            self.unskew(i, j + 1),
            self.unskew(i + 1, j + 1),
        ];

        let (mut x_lo, mut x_hi, mut y_lo, mut y_hi) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for [x, y] in corners {
            x_lo = x_lo.min(x);
            x_hi = x_hi.max(x);
            y_lo = y_lo.min(y);
            y_hi = y_hi.max(y);
        }

        let gw = self.grid_size[0] as i32;
        let gh = self.grid_size[1] as i32;
        let ox_lo = (((x_lo - self.origin[0]) / self.increment[0]).floor() as i32).max(0);
        let ox_hi = (((x_hi - self.origin[0]) / self.increment[0]).ceil() as i32).min(gw);
        let oy_lo = (((y_lo - self.origin[1]) / self.increment[1]).floor() as i32).max(0);
        let oy_hi = (((y_hi - self.origin[1]) / self.increment[1]).ceil() as i32).min(gh);

        if ox_lo >= ox_hi || oy_lo >= oy_hi {
            None
        } else {
            Some([ox_lo, ox_hi, oy_lo, oy_hi])
        }
    }
}