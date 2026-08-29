use std::f32::consts::SQRT_2;

use std::mem::MaybeUninit;

use simply_simd::{ Arch, Simd, enable_targets };

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{ Combiner, CombinerState };
use crate::noise::util::grid_data::{ GridData};
use crate::noise::util::grid_helpers::{
    Arena,
    ArenaBuffer,
    MaybeUninitSliceSimdExt,
    maybe_tail_load,
    maybe_tail_store,
    pad_grid_size,
    validate_grid_size,
    validate_state_size,
    simd_rem_euclid_i32,
};
use crate::{ Simplex, GridGenerator };


const SQRT_3: f32 = 1.732_050_8;
const SKEW_2D: f32 = (SQRT_3 - 1.0) / 2.0;
const UNSKEW_2D: f32 = (3.0 - SQRT_3) / 6.0;

const SCALE: f32 = 80.0;
const SCALED_SQRT: f32 = (SQRT_2 / 2.0) * SCALE;

const A: f32 = SCALE;
const B: f32 = SCALED_SQRT;
const C: f32 = 0.0;
pub const X_GRADIENTS_2D: [f32; 8] = [A, B, C, -B, -A, -B, C, B];
pub const Y_GRADIENTS_2D: [f32; 8] = [C, B, A, B, C, -B, -A, -B];

const BYTE_SHUFFLE: [u8; 64] = [
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13,
];

struct RowWindow {
    top: *mut f32,
    bot: *mut f32,
    width: usize,
}

impl RowWindow {
    fn new(arena: &mut Arena, width: usize) -> Self {
        Self {
            top: arena
                .allocate::<f32>(width * 2)
                .as_mut_ptr()
                .cast(),
            bot: arena
                .allocate::<f32>(width * 2)
                .as_mut_ptr()
                .cast(),
            width,
        }
    }

    #[inline(always)]
    fn fill_row<A: Arch>(
        params: &GridNoiseParams<2>,
        grid_data: &GridData<2>,
        buff: &mut [(f32, f32)],
        ly: i32,
        lx_start: i32
    ) {
        let ly = grid_data.octave_tiling[1].map_or(ly, |t| ly.rem_euclid(t as i32));
        let y_shuf = hash_cell_y::<A>(ly as u32, params.seed);
        let y_shuf_v = Simd::<u32, A>::splat(y_shuf);
        let lanes = Simd::<f32, A>::LANES;

        let mut x_it = 0;
        while x_it + lanes <= buff.len() {
            let lx_base = lx_start.wrapping_add(x_it as i32);
            let lx_v = Simd::<i32, A>::splat(lx_base) + Simd::<i32, A>::iota(0);
            let lx_v = if let Some(t) = grid_data.octave_tiling[0] {
                simd_rem_euclid_i32::<A>(lx_v, t as i32)
            } else {
                lx_v
            };
            let hashes = hash_cells_row::<A>(lx_v.raw_cast(), y_shuf_v, params.seed);
            let indices = hashes >> Simd::<u32, A>::splat(29);
            let gx = indices.gather(&X_GRADIENTS_2D);
            let gy = indices.gather(&Y_GRADIENTS_2D);
            let gx_arr = gx.to_array();
            let gy_arr = gy.to_array();
            for i in 0..lanes {
                buff[x_it + i] = (gx_arr[i], gy_arr[i]);
            }
            x_it += lanes;
        }
        for i in x_it..buff.len() {
            let lx = lx_start.wrapping_add(i as i32);
            let lx = grid_data.octave_tiling[0].map_or(lx, |t| lx.rem_euclid(t as i32));
            let idx = (hash_cell_with_y::<A>(lx as u32, y_shuf, params.seed) >> 29) as usize;
            buff[i] = (X_GRADIENTS_2D[idx], Y_GRADIENTS_2D[idx]);
        }
    }

    fn top(&self) -> &[(f32, f32)] {
        unsafe { std::slice::from_raw_parts(self.top.cast::<(f32, f32)>(), self.width) }
    }

    fn bot(&self) -> &[(f32, f32)] {
        unsafe { std::slice::from_raw_parts(self.bot.cast::<(f32, f32)>(), self.width) }
    }

    fn top_mut(&mut self) -> &mut [(f32, f32)] {
        unsafe { std::slice::from_raw_parts_mut(self.top.cast::<(f32, f32)>(), self.width) }
    }

    fn bot_mut(&mut self) -> &mut [(f32, f32)] {
        unsafe { std::slice::from_raw_parts_mut(self.bot.cast::<(f32, f32)>(), self.width) }
    }

    #[inline(always)]
    fn swap(&mut self) {
        std::mem::swap(&mut self.top, &mut self.bot);
    }
}

/// Hashes `LANES` consecutive lattice columns from a pre-built `lx_v` vector
/// at fixed y in one shot
#[inline(always)]
fn hash_cells_row<A: Arch>(lx_v: Simd<u32, A>, y_shuf: Simd<u32, A>, seed: u32) -> Simd<u32, A> {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(0x85ebca6b_u32);
    let seed_v = Simd::<u32, A>::splat(seed);

    let x_shuf = (lx_v * seed_v).permute_8(shuffle_indices) ^ prime;
    (x_shuf * y_shuf) ^ x_shuf
}

#[inline(always)]
fn hash_cell_y<A: Arch>(y: u32, seed: u32) -> u32 {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(0x85ebca6b_u32);
    (Simd::<u32, A>::splat(y.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime).to_array()[0]
}

#[inline(always)]
fn hash_cell_with_y<A: Arch>(x: u32, y_shuf: u32, seed: u32) -> u32 {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(0x85ebca6b_u32);
    let x_shuf = (
        Simd::<u32, A>::splat(x.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];
    x_shuf.wrapping_mul(y_shuf) ^ x_shuf
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
        validate_state_size::<C, A, _>(params.grid_size, state.len());
        let padded_size = pad_grid_size::<A, _>(params.grid_size);

        let required_cache =
            padded_size[1] * 3 + padded_size[0] * 3 + (padded_size[0] + 1) * 8 + 24;
        let mut cache = ArenaBuffer::<A>::with_capacity(required_cache);
        let mut arena = Arena::with_cache(&mut cache);
        let mut sub_arena = arena.allocate_arena(padded_size[0] * 3 + padded_size[1] * 3);

        let grid_data = GridData::new::<A>(&params, &mut sub_arena, &padded_size);

        let row_len = grid_data.num_loops[0] + 3;
    }
}