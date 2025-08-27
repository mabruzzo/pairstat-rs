use crate::bins::BinEdges;
use crate::misc::segment_idx_bounds;
use crate::parallel::{ReductionSpec, StandardTeamParam, Team};
use crate::reducer::{Datum, Reducer};
use crate::state::StatePackViewMut;
use crate::twopoint::{
    common::PairOperation,
    idx_3d_offset::{Idx3DOffsetSeq, get_block_a_start_stop_indices},
    spatial::{CartesianBlock, CellWidth},
};

/// Computes contributions to binned statistics from values computed from the
/// specified pairs of points.
pub struct TwoPointCartesian<'a, R: Reducer, B: BinEdges> {
    reducer: R,
    block_a: CartesianBlock<'a>,
    // block_b may reference the same arrays as block_a
    block_b: CartesianBlock<'a>,
    cell_width: CellWidth,
    // todo: consider storing distance_bin_edges
    squared_distance_bin_edges: B,
    idx3offset_seq: Idx3DOffsetSeq,
    pair_op: PairOperation,
}

impl<'a, R: Reducer, B: BinEdges> TwoPointCartesian<'a, R, B> {
    /// construct a new instance
    pub fn new(
        reducer: R,
        block_a: CartesianBlock<'a>,
        block_b: Option<CartesianBlock<'a>>,
        cell_width: CellWidth,
        squared_distance_bin_edges: B,
        pair_op: PairOperation,
    ) -> Result<Self, &'static str> {
        // todo: consider storing distance_bin_edges
        let (idx3offset_seq, other_block) = match block_b {
            Some(block_b) => (Idx3DOffsetSeq::new_cross(&block_a, &block_b)?, block_b),
            None => (Idx3DOffsetSeq::new_auto(&block_a)?, block_a.clone()),
        };
        if idx3offset_seq.is_empty() {
            // I think this can only happen when block_b was originally None
            Err("the provided blocks have no pairs")
        } else {
            Ok(Self {
                reducer,
                block_a,
                block_b: other_block,
                cell_width,
                squared_distance_bin_edges,
                idx3offset_seq,
                pair_op,
            })
        }
    }
}

impl<'a, R: Reducer, B: BinEdges> ReductionSpec for TwoPointCartesian<'a, R, B> {
    type ReducerType = R;
    /// return a reference to the reducer
    fn get_reducer(&self) -> &Self::ReducerType {
        &self.reducer
    }

    /// The number of bins in this reduction.
    fn n_bins(&self) -> usize {
        self.squared_distance_bin_edges.n_bins()
    }

    fn team_loop_bounds(&self, team_id: usize, _team_info: &StandardTeamParam) -> (usize, usize) {
        (team_id, team_id + 1)
    }

    const NESTED_REDUCE: bool = false;

    fn add_contributions<T: Team>(
        &self,
        binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
        team_loop_index: usize,
        team: &mut T,
    ) {
        debug_assert_eq!(team_loop_index, team.team_id());

        match &self.pair_op {
            PairOperation::ElementwiseMultiply => {
                perform_reduce::<false, T>(
                    binned_statepack,
                    team,
                    &self.reducer,
                    &self.block_a,
                    &self.block_b,
                    &self.cell_width,
                    &self.squared_distance_bin_edges,
                    &self.idx3offset_seq,
                );
            }
            PairOperation::ElementwiseSub => {
                perform_reduce::<true, T>(
                    binned_statepack,
                    team,
                    &self.reducer,
                    &self.block_a,
                    &self.block_b,
                    &self.cell_width,
                    &self.squared_distance_bin_edges,
                    &self.idx3offset_seq,
                );
            }
        }
    }
}

/// does the heavy lifting for [`TwoPointCartesian`]
///
/// This logic exists outside of [`TwoPointCartesian::add_contributions`]
/// for 2 reasons:
/// 1. to make `SUBTRACT` a const-generic
/// 2. to reduce the number of indentations
#[allow(clippy::too_many_arguments)]
fn perform_reduce<const SUBTRACT: bool, T: Team>(
    binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
    team: &mut T,
    reducer: &impl Reducer,
    block_a: &CartesianBlock,
    block_b: &CartesianBlock,
    cell_width: &CellWidth,
    squared_distance_bin_edges: &impl BinEdges,
    idx3offset_seq: &Idx3DOffsetSeq,
) {
    let team_info = team.standard_team_info();
    let (start, stop) = segment_idx_bounds(idx3offset_seq.len(), team.team_id(), team_info.n_teams);

    for seq_index in start..stop {
        // during the current iteration, team members collectively compute the
        // contribution to the overall binned reduction of every
        // measurement-pair from block_a & block_b described by idx_3d_offset
        let idx_3d_offset = idx3offset_seq.get(seq_index);
        // recall: "adding" idx_3d_offset to a 3D index from block_a produces
        // the index of the block_b measurement that is part of the pair

        // let's determine the common distance bin that every one of these
        // pairs (considered in the current iteration) contributes to
        let dist2 = idx_3d_offset.distance_squared(block_a, block_b, cell_width);
        let Some(bin_idx) = squared_distance_bin_edges.bin_index(dist2) else {
            continue; // dist2 doesn't correspond to a distance bin
        };

        // get loop bounds to visit the 3D index of each block_a measurement
        // that is part of a measurement-pair (described by idx_3d_offset)
        // with a block_b measurement
        let ([k_start, j_start, i_start], [k_stop, j_stop, i_stop]) =
            get_block_a_start_stop_indices(&idx_3d_offset, block_a, block_b);

        // extract the increments that we'll add to a 3d index from block_a
        // to get a pair's corresponding index for block_b
        // -> NOTE: its tempting to consider that we could just add a 1D
        //    offset, but that gets **REALLY** messy in the general case where
        //    block_a and block_b to have different shapes/strides
        let [dk, dj, di] = idx_3d_offset.value();

        // define the logic that each team member is responsible for
        if T::IS_VECTOR_PROCESSOR {
            // (things need to be a little different to support SIMD)
            panic!("can't handle this case! (yet)");
        }
        let f = |tmp_accum_states: &mut StatePackViewMut, member_id: usize| {
            // prepare the tmp_accum_state where the member adds contributions
            debug_assert_eq!(tmp_accum_states.n_states(), 1); // sanity check!
            let mut tmp_accum_state = tmp_accum_states.get_state_mut(0);

            // determine the i indices that the current member considers
            let adjusted_i_start = i_start + (member_id as isize);
            let i_step = team_info.n_members_per_team;

            // the current loop structure is a little inefficient. Currently
            // there may be idle members. The number of idle members is:
            //   `max(0, team_info.n_members_per_team - (i_stop - i_start))`
            for k in k_start..k_stop {
                for j in j_start..j_stop {
                    // TODO confirm that step_by doesn't trip up the GPU (compare to while-loop)
                    for i in (adjusted_i_start..i_stop).step_by(i_step) {
                        let i_a = block_a.idx_spec.map_idx3d_to_1d(k, j, i) as usize;
                        let va_x = block_a.value_components_zyx[2][i_a];
                        let va_y = block_a.value_components_zyx[1][i_a];
                        let va_z = block_a.value_components_zyx[0][i_a];
                        let wa = block_a.weights[i_a];

                        let i_b = block_b.idx_spec.map_idx3d_to_1d(k + dk, j + dj, i + di) as usize;
                        let vb_x = block_b.value_components_zyx[2][i_b];
                        let vb_y = block_b.value_components_zyx[1][i_b];
                        let vb_z = block_b.value_components_zyx[0][i_b];
                        let wb = block_b.weights[i_b];

                        let datum = Datum {
                            value: if SUBTRACT {
                                [vb_z - va_z, vb_y - va_y, vb_x - va_x]
                            } else {
                                [vb_z * va_z, vb_y * va_y, vb_x * va_x]
                            },
                            weight: wa * wb,
                        };
                        reducer.consume(&mut tmp_accum_state, &datum);
                    } // end i-loop
                } // end j-loop
            } // end k-loop
        };

        // the next function does the following:
        // 1. each member invokes f and stores the contributions from the
        //    measurement pairs that the member was responsible for to a
        //    temporary `accum_state` buffer provided by `team`
        // 2. uses `reducer` to merge these `accum_state` buffers
        // 3. has a single member use that buffer and `reducer` to update the
        //    `accum_state` in `binned_statepack` corresponding to `bin_idx`
        team.calccontribs_combine_apply(binned_statepack, reducer, bin_idx, &f);
    }
}
