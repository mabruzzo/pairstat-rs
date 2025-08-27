// this file is intended for illustrative purposes
// see module-level documentation for chunked.rs for more detail

use crate::misc::segment_idx_bounds;
use crate::parallel::{BinnedDatum, ReductionSpec, StandardTeamParam, Team};
use crate::reduce_sample::chunked::{QuadraticPolynomial, SampleDataStreamView};
use crate::reducer::{Comp0Mean, Datum, Reducer};
use crate::state::StatePackViewMut;

// Version 0: our simple naive implementation
pub fn naive_mean_unordered(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    tot_weight_bins: &mut [f64],
    mean_times_weight_bins: &mut [f64],
) {
    let n_bins = tot_weight_bins.len();
    assert_eq!(n_bins, mean_times_weight_bins.len());

    for i in 0..stream.len() {
        let bin_index = stream.bin_indices[i];
        let x = stream.x_array[i];
        let w = stream.weights[i];
        if bin_index < n_bins {
            tot_weight_bins[bin_index] += w;
            mean_times_weight_bins[bin_index] += w * f.call(x);
        }
    }
}

// Version 1: an implementation that uses accumulator machinery
pub fn reducer_mean_unordered(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    reducer: &Comp0Mean,
    binned_statepack: &mut StatePackViewMut,
) {
    assert_eq!(reducer.accum_state_size(), binned_statepack.state_size());
    let n_bins = binned_statepack.n_states();

    for i in 0..stream.len() {
        let bin_index = stream.bin_indices[i];
        let datum = Datum::from_scalar_value(f.call(stream.x_array[i]), stream.weights[i]);
        if bin_index < n_bins {
            reducer.consume(&mut binned_statepack.get_state_mut(bin_index), &datum);
        }
    }
}

// TODO Delete everything in the giant multi-line comment
//      (we need to ensure the file still makes sense after the fact or
//      finish PR #59)
/*
// At this point, we will introduce a new version of the function that
// rearranges the work in a manner that would be suitable for parallelization.
//
// For our purposes, we define 2 levels of parallelized processing:
// 1. a fine-grained synchronous/collaborative level where the instructions
//    are applied to multiple values that are packed together in memory in a
//    synchronized manner
//    - on a CPU, this parallelism is achieved with SIMD instructions
//    - on a GPUs, these operations are performed generally performed by
//      threads grouped in a Warp (AMD calls them Wave Fronts)
//    - technically GPU threads in a Warp have a lot more flexibility, but
//      they perform extremely well in these contexts. But we target this sort
//      of parallelism for maximum portability.
// 2. a coarser-grained parallelism where separate threads operate much more
//    independently from each other.
//
// We will focus on rewriting the function in a manner conducive to the first
// type and afterwards, we'll write another version conducive to both types

// Version 2: In this version we organize the work in a manner conducive to
// fine-grained synchronous parallel processing.
// - we won't actually achieve this kind of processing in this function, but
//   we'll explain how we might do it
// - in a later version of this functionality, we will worry about parallelism
//
// Motivating our design.
// - ideally, we would love to somehow parallelize the whole process of
//   generating the (datum, bin-index) pairs from the data source, and using
//   it to update the appropriate accum_state bins.
// - Unfortunately, the unpredictable order is a large obstacle to
//   parallelizing the update of accumulator state (in a general way)
// - So, we do the best we possibly can. We treat the task of generating the
//   (datum, bin-index) pairs separately from updating the appropriate
//   accumulator state. This is a batching strategy:
//   - pre-generate a batch of pairs and store them in a collection-pad
//     (this operation can be parallelized)
//   - then we update the accumulator from the batch of pairs (this must be
//     sequential)
//
// Because this function is defined in a no_std, the collection-pad is
// pre-allocated.
//
// It also accepts an optional tuple holding the starting & stopping indices to
// consider from the stream (more on that later)
pub fn restructured1_mean_unordered(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    reducer: &Comp0Mean,
    binned_statepack: &mut StatePackViewMut,
    collect_pad: &mut [BinnedDatum],
    start_stop_idx: Option<(usize, usize)>,
) {
    let batch_size = collect_pad.len();
    assert!(batch_size > 0);
    assert_eq!(reducer.accum_state_size(), binned_statepack.state_size());
    let n_bins = binned_statepack.n_states();

    let (i_start, i_stop) = start_stop_idx.unwrap_or((0, stream.len()));
    assert!(i_stop >= i_start); // when they are equal, we do no work

    let n_batches = (i_stop - i_start).div_ceil(batch_size);
    for batch_idx in 0..n_batches {
        // While we execute the next loop sequentially, we can parallelize it:
        // - on CPUs, it can be vectorized
        // - on GPUs, threads will work together
        #[allow(clippy::needless_range_loop)]
        for pad_idx in 0..batch_size {
            let i = i_start + batch_idx * batch_size + pad_idx;

            collect_pad[pad_idx] = if i == i_stop {
                // NOTE: we can process an arbitrary number of zeroed data
                //       elements without affecting the output since it holds
                //       a weight of 0
                BinnedDatum::zeroed()
            } else {
                BinnedDatum {
                    bin_index: stream.bin_indices[i],
                    datum: Datum::from_scalar_value(f.call(stream.x_array[i]), stream.weights[i]),
                }
            };
        }

        // now we sequentially process the (datum, bin-index) pairs
        // -> there isn't a general, portable way to do this
        for e in collect_pad.iter() {
            let bin_index = e.bin_index;
            let datum = &e.datum;

            if bin_index < n_bins {
                reducer.consume(&mut binned_statepack.get_state_mut(bin_index), datum);
            }
        }
    }
}

// now we finally write a function that carves up the task in a way conducive
// to achieving the coarser grained parallelism.
//
// Essentially, the strategy is to cut up the index-range into segment.
// - we accumulate the contributions of each segment in a separate temporary
//   binned_statepack (we use `restructured1_mean_unordered` to actually gather these
//   contributions)
// - at the very end, we gather up all the contributions and use them to update
//   the output binned_statepack
//
// scratch_statepack_storage is provided to simulate the fact that each thread
pub fn restructured2_mean_unordered(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    reducer: &Comp0Mean,
    binned_statepack: &mut StatePackViewMut,
    scratch_binned_statepacks: &mut [StatePackViewMut],
    collect_pad: &mut [BinnedDatum],
) {
    // a scratch_binned_statepack has been provided for every segment
    let n_segments = scratch_binned_statepacks.len();
    assert!(n_segments > 0);

    // each pass through this loop considers a separate index-segment.
    // -> this work could be distributed among different threads (or thread
    //    groups)
    // -> if we did parallelize this loop, each thread (or thread group) would
    //    need to allocate its own collect_pad
    #[allow(clippy::needless_range_loop)]
    for seg_index in 0..n_segments {
        let local_binned_statepack = &mut scratch_binned_statepacks[seg_index];
        reset_full_statepack(reducer, local_binned_statepack);

        // determine DataStream indices to be processed in the current segment
        let idx_bounds = segment_idx_bounds(stream.len(), seg_index, n_segments);

        // now do the work!
        restructured1_mean_unordered(
            stream,
            f,
            reducer,
            local_binned_statepack,
            collect_pad,
            Some(idx_bounds),
        );
    }

    // after all the above loop is done, let's merge together our results such
    // that scratch_binned_statepacks[0] holds all the contributions
    serial_consolidate_scratch_statepacks(reducer, scratch_binned_statepacks);

    // finally add the contribution to binned_statepack
    merge_full_statepacks(
        reducer,
        binned_statepack,
        &scratch_binned_statepacks.first().unwrap().as_view(),
    );
}
*/

pub struct MeanUnorderedReduction<'a> {
    stream: SampleDataStreamView<'a>,
    f: QuadraticPolynomial,
    reducer: Comp0Mean,
    n_bins: usize,
}

impl<'a> MeanUnorderedReduction<'a> {
    pub fn new(
        stream: SampleDataStreamView<'a>,
        f: QuadraticPolynomial,
        reducer: Comp0Mean,
        n_bins: usize,
    ) -> Self {
        Self {
            stream,
            f,
            reducer,
            n_bins,
        }
    }
}

impl<'a> ReductionSpec for MeanUnorderedReduction<'a> {
    type ReducerType = Comp0Mean;

    fn get_reducer(&self) -> &Self::ReducerType {
        &self.reducer
    }

    fn n_bins(&self) -> usize {
        self.n_bins
    }

    fn team_loop_bounds(&self, team_id: usize, team_info: &StandardTeamParam) -> (usize, usize) {
        let stream_idx_bounds = segment_idx_bounds(self.stream.len(), team_id, team_info.n_teams);
        let n_stream_indices = stream_idx_bounds.1 - stream_idx_bounds.0;
        let batch_size = team_info.n_members_per_team;
        let n_batches = n_stream_indices.div_ceil(batch_size);
        (0, n_batches)
    }

    const NESTED_REDUCE: bool = false;

    fn add_contributions<T: Team>(
        &self,
        binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
        team_loop_index: usize,
        team: &mut T,
    ) {
        let team_id = team.team_id();
        let team_param = team.standard_team_info();
        let stream_idx_bounds = segment_idx_bounds(self.stream.len(), team_id, team_param.n_teams);
        let i_offset = stream_idx_bounds.0 + team_param.n_members_per_team * team_loop_index;

        let stream_len = self.stream.len();

        if T::IS_VECTOR_PROCESSOR {
            team.collect_pairs_then_apply(
                binned_statepack,
                self.get_reducer(),
                &|collect_pad: &mut [BinnedDatum], member_id: usize| {
                    assert_eq!(member_id, 0); // sanity check
                    assert_eq!(collect_pad.len(), team_param.n_members_per_team); // sanity check!
                    // we would need to do a lot of work to get this to
                    // auto-vectorize
                    // - we probably need to pre-generate more elements than
                    //   there are team members.
                    // - we probably need to statically make guarantees about
                    //   alignment and array length
                    // - it may be easier to use machinery like glam::DVec to
                    //   force the vectorizaiton

                    #[allow(clippy::needless_range_loop)]
                    for lane_id in 0..team_param.n_members_per_team {
                        let i = i_offset + lane_id;
                        collect_pad[lane_id] = if i >= stream_len {
                            BinnedDatum::zeroed()
                        } else {
                            BinnedDatum {
                                bin_index: self.stream.bin_indices[i],
                                datum: Datum::from_scalar_value(
                                    self.f.call(self.stream.x_array[i]),
                                    self.stream.weights[i],
                                ),
                            }
                        };
                    }
                },
            );
        } else {
            team.collect_pairs_then_apply(
                binned_statepack,
                self.get_reducer(),
                &|collect_pad: &mut [BinnedDatum], member_id: usize| {
                    assert_eq!(collect_pad.len(), 1); // sanity check!
                    let i = i_offset + member_id;
                    collect_pad[0] = if i >= stream_len {
                        BinnedDatum::zeroed()
                    } else {
                        BinnedDatum {
                            bin_index: self.stream.bin_indices[i],
                            datum: Datum::from_scalar_value(
                                self.f.call(self.stream.x_array[i]),
                                self.stream.weights[i],
                            ),
                        }
                    };
                },
            );
        }
    }
}
