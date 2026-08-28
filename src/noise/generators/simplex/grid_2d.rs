use std::f32::consts::SQRT_2;

use std::mem::MaybeUninit;

use simply_simd::{ Arch, Simd, enable_targets };

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{ Combiner, CombinerState };
use crate::noise::util::grid_data::{ GridData, Lerp };
use crate::noise::util::grid_helpers::{
    Arena,
    ArenaBuffer,
    MaybeUninitSliceSimdExt,
    maybe_tail_load,
    maybe_tail_store,
    pad_grid_size,
    validate_grid_size,
    validate_state_size,
};
use crate::{ Simplex, GridGenerator };

const LERP: u8 = Lerp::Quintic as u8;

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

#[enable_targets(A)]
impl GridGenerator<2> for Simplex {
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<2>,
        combiner: C::Config,
        state: &mut [f32],
        dst: &mut [f32]
    ) {

    }
}