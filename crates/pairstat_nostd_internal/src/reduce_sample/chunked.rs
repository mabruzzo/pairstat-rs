//! This crate is primarily intended to define a simple example of how the
//! reduction-parallelism functionality works. We don't currently intend to
//! expose anything defined here as part of the public API.
//!
//! This serves 2 purposes:
//! 1. a pedagogical resource for learning/teaching the underlying machinery
//! 2. simplified logic we can use for easy unit-testing
//!
//! # Overview
//! This crate effectively solves a simple problem with increasingly
//! sophisticated solutions that are representative of how the reduction
//! machinery is intended to be used.
//!
//! # The Problem
//! Suppose we have a stream of data, where each datum is composed of a
//! precomputed bin-index, `i`, a value, `x`, and a weight `w`. For some
//! function `f`, we want to compute the weighted average value of `f(x)`.
//!
//! We consider 2 versions of this problem.
//! -> in this file, we consider the scenario where we have chunks of data
//!    with a known bin-index
//! -> in unordered.rs, we consider a variation where we assume that the
//!    data is in a totally random order
//! The logic has been separated between files to facilitate side-by-side
//! comparisons

use crate::misc::segment_idx_bounds;
use crate::parallel::{ReductionSpec, StandardTeamParam, Team};
use crate::reducer::{Comp0Mean, Datum, Reducer};
use crate::state::{AccumStateViewMut, StatePackViewMut};

// Defining some basic functionality for implementing this example:
// ================================================================

// for simplicity, we assume that the function that operates on the data stream
// can always be modelled as a quadratic polynomial (this assumption is
// primarily made to simplify the code)

/// Models the quadratic polynomial: `a*x^2 + b*x + c`
#[derive(Clone, Copy)]
pub struct QuadraticPolynomial {
    a: f64,
    b: f64,
    c: f64,
}

impl QuadraticPolynomial {
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }

    pub fn call(&self, x: f64) -> f64 {
        self.c + x * (self.b + x * self.a)
    }
}

#[derive(Clone)]
pub struct SampleDataStreamView<'a> {
    // we only make the members public so that they can get reused in unordered.rs
    pub bin_indices: &'a [usize],
    pub x_array: &'a [f64],
    pub weights: &'a [f64],
    pub chunk_lens: Option<&'a [usize]>,
}

impl<'a> SampleDataStreamView<'a> {
    // ignore the chunk_lens argument
    pub fn new(
        bin_indices: &'a [usize],
        x_array: &'a [f64],
        weights: &'a [f64],
        chunk_lens: Option<&'a [usize]>,
    ) -> Result<Self, &'static str> {
        let len = bin_indices.len();

        if chunk_lens.is_some_and(|arr| len != arr.iter().sum()) {
            Err("the values in chunk_lens don't add up the bin_indices.len()")
        } else if (len != x_array.len()) || (len != weights.len()) {
            Err("the lengths of the slices are NOT consistent")
        } else {
            Ok(SampleDataStreamView {
                bin_indices,
                x_array,
                weights,
                chunk_lens,
            })
        }
    }

    pub fn first_index_of_chunk(&self, i: usize) -> Result<usize, &'static str> {
        if let Some(chunk_lens) = &self.chunk_lens {
            if i >= chunk_lens.len() {
                Err("i exceeds the number of chunks")
            } else {
                Ok(chunk_lens.iter().take(i).sum())
            }
        } else {
            Err("the stream doesn't have chunks")
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bin_indices.len()
    }
}

// Version 0: our simple naive implementation
pub fn naive_mean_chunked(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    tot_weight_bins: &mut [f64],
    mean_times_weight_bins: &mut [f64],
) {
    let n_bins = tot_weight_bins.len();
    assert_eq!(n_bins, mean_times_weight_bins.len());

    let Some(chunk_lens) = stream.chunk_lens else {
        panic!("the data stream wasn't chunked!")
    };

    let mut i_global = 0;
    for chunk_len in chunk_lens.iter().cloned() {
        let bin_index = stream.bin_indices[i_global];
        if bin_index < n_bins {
            let mut local_tot_weight = 0.0;
            let mut tmp_mean_times_weight = 0.0;
            for i in i_global..(i_global + chunk_len) {
                local_tot_weight += stream.weights[i];
                tmp_mean_times_weight += stream.weights[i] * f.call(stream.x_array[i]);
            }
            tot_weight_bins[bin_index] += local_tot_weight;
            mean_times_weight_bins[bin_index] += tmp_mean_times_weight;
        }
        i_global += chunk_len
    }
}

// Version 1: an implementation that uses accumulator machinery
pub fn reducer_mean_chunked(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    reducer: &Comp0Mean,
    binned_statepack: &mut StatePackViewMut,
) {
    let Some(chunk_lens) = stream.chunk_lens else {
        panic!("the data stream wasn't chunked!")
    };

    let n_bins = binned_statepack.n_states();
    assert_eq!(reducer.accum_state_size(), binned_statepack.state_size());
    assert_eq!(reducer.accum_state_size(), 2);
    let mut tmp_buffer = [0.0; 2];
    let mut tmp_accum_state = AccumStateViewMut::from_contiguous_slice(&mut tmp_buffer);

    let mut i_global = 0;
    for chunk_len in chunk_lens.iter().cloned() {
        let bin_index = stream.bin_indices[i_global];
        if bin_index < n_bins {
            reducer.init_accum_state(&mut tmp_accum_state);
            for i in i_global..(i_global + chunk_len) {
                reducer.consume(
                    &mut tmp_accum_state,
                    &Datum::from_scalar_value(f.call(stream.x_array[i]), stream.weights[i]),
                );
            }
            reducer.merge(
                &mut binned_statepack.get_state_mut(bin_index),
                &tmp_accum_state.as_view(),
            );
        }
        i_global += chunk_len
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

/// Version 2: In this version we organize the work in a manner conducive to
/// fine-grained synchronous parallel processing.
/// - we won't actually achieve this kind of processing in this function, but
///   we'll explain how we might do it
/// - in a later version of this functionality, we will worry about parallelism
///
/// In this case, we take advantage of the fact that we know that we have chunks
/// of data with a known bin-index.
/// - Thus, we can apply a nested reduction on all of the entries in a chunk
///   before we add the total result
///
/// Because this function is defined in a `#![no_std]` environment, the buffers
/// used for holding the temporary accum_states (used in the nested reduction)
/// have been pre-allocated and provided in the `tmp_accum_states` variable.
///
/// It also accepts an optional tuple holding the starting & stopping indices to
/// consider from the stream (more on that later)
pub fn restructured1_mean_chunked(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    reducer: &Comp0Mean,
    binned_statepack: &mut StatePackViewMut,
    tmp_accum_states: &mut StatePackViewMut,
    start_stop_chunk_idx: Option<(usize, usize)>,
) {
    let Some(chunk_lens) = stream.chunk_lens else {
        panic!("the data stream wasn't chunked!")
    };

    let n_bins = binned_statepack.n_states();
    assert_eq!(reducer.accum_state_size(), binned_statepack.state_size());
    assert_eq!(reducer.accum_state_size(), tmp_accum_states.state_size());

    let n_tmp_accum_states = tmp_accum_states.n_states();

    let (chunk_i_start, chunk_i_stop) = start_stop_chunk_idx.unwrap_or((0, chunk_lens.len()));
    let mut i_global = stream.first_index_of_chunk(chunk_i_start).unwrap();
    #[allow(clippy::needless_range_loop)]
    for chunk_i in chunk_i_start..chunk_i_stop {
        let chunk_len = chunk_lens[chunk_i];
        let bin_index = stream.bin_indices[i_global];
        if bin_index < n_bins {
            // reset the the state in each of our temporary states
            reset_full_statepack(reducer, tmp_accum_states);
            // collect contributions in each of our temporary states
            // - currently, we gather all contributions for a single
            //   tmp_accum_state and then we move on to the next
            //   tmp_accum_state
            //
            // to achieve auto-vectorization:
            // - we'd probably need to massage the way this loop is implemented
            //   so that we are updating the states of all tmp_accum_states in
            //   a single operation.
            //   - the easiest way to do this probably involves modifying
            //     things so that an tmp_accum_state is implemented in terms
            //     of something like a glam::DVec type and so that the accum
            //     methods can operate on glam::DVec objects
            //   - this would involve some refactoring and statically encoding
            //     some assumptions about the alignment of data storage
            // - note: this change would only involve modifying implementation,
            //   we wouldn't actually alter the algorithm
            for offset in 0..n_tmp_accum_states {
                let mut tmp_accum_state = tmp_accum_states.get_state_mut(offset);
                let i_itr =
                    ((i_global + offset)..(i_global + chunk_len)).step_by(n_tmp_accum_states);
                for i in i_itr {
                    reducer.consume(
                        &mut tmp_accum_state,
                        &Datum::from_scalar_value(f.call(stream.x_array[i]), stream.weights[i]),
                    );
                }
            }
            // now, we merge together the temporary accumulator states
            // tmp_accum_states.get_state(0) now holds the total contribution
            serial_merge_accum_states(reducer, tmp_accum_states);

            reducer.merge(
                &mut binned_statepack.get_state_mut(bin_index),
                &tmp_accum_states.get_state(0),
            );
        }
        i_global += chunk_len
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
pub fn restructured2_mean_chunked(
    stream: &SampleDataStreamView,
    f: QuadraticPolynomial,
    reducer: &Comp0Mean,
    binned_statepack: &mut StatePackViewMut,
    scratch_binned_statepacks: &mut [StatePackViewMut],
    tmp_accum_states: &mut StatePackViewMut,
) {
    let Some(chunk_lens) = stream.chunk_lens else {
        panic!("the data stream wasn't chunked!")
    };

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

        // determine chunk indices to be processed in the current segment
        let chunk_idx_bounds = segment_idx_bounds(chunk_lens.len(), seg_index, n_segments);

        // now do the work!
        restructured1_mean_chunked(
            stream,
            f,
            reducer,
            local_binned_statepack,
            tmp_accum_states,
            Some(chunk_idx_bounds),
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

pub struct MeanChunkedReduction<'a> {
    stream: SampleDataStreamView<'a>,
    stream_chunk_lens: &'a [usize],
    f: QuadraticPolynomial,
    reducer: Comp0Mean,
    n_bins: usize,
}

impl<'a> MeanChunkedReduction<'a> {
    pub fn new(
        stream: SampleDataStreamView<'a>,
        f: QuadraticPolynomial,
        reducer: Comp0Mean,
        n_bins: usize,
    ) -> Self {
        let Some(chunk_lens) = stream.chunk_lens else {
            panic!("the data stream wasn't chunked!")
        };
        Self {
            stream,
            stream_chunk_lens: chunk_lens,
            f,
            reducer,
            n_bins,
        }
    }
}

impl<'a> ReductionSpec for MeanChunkedReduction<'a> {
    type ReducerType = Comp0Mean;

    fn get_reducer(&self) -> &Self::ReducerType {
        &self.reducer
    }

    fn n_bins(&self) -> usize {
        self.n_bins
    }

    fn team_loop_bounds(&self, team_id: usize, team_info: &StandardTeamParam) -> (usize, usize) {
        segment_idx_bounds(self.stream_chunk_lens.len(), team_id, team_info.n_teams)
    }

    const NESTED_REDUCE: bool = true;

    fn add_contributions<T: Team>(
        &self,
        binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
        team_loop_idx: usize,
        team: &mut T,
    ) {
        // figure out the first index of the stream that corresponds to the
        // current chunk, the associated bin_index, and the chunk_len
        let chunk_index = team_loop_idx;
        let i_global = self.stream.first_index_of_chunk(chunk_index).unwrap();
        let bin_index = self.stream.bin_indices[i_global];
        if bin_index >= self.n_bins {
            return;
        }
        let chunk_len = self.stream_chunk_lens[chunk_index];

        // get the team info
        let team_param = team.standard_team_info();

        if T::IS_VECTOR_PROCESSOR {
            team.calccontribs_combine_apply(
                binned_statepack,
                &self.reducer,
                bin_index,
                &|tmp_accum_states: &mut StatePackViewMut, member_id: usize| {
                    // to achieve auto-vectorization:
                    // - we'd probably need to massage the way this loop is
                    //   implemented so that we are updating the states of all
                    //   tmp_accum_states in a single operation.
                    //   - the easiest way to do this probably involves
                    //     modifying things so that a tmp_accum_state is
                    //     implemented in terms of something like a
                    //     glam::DVec type and so that the Reducer methods
                    //     can operate on glam::DVec objects
                    //   - this would involve some refactoring and statically
                    //     encoding some assumptions about the alignment of
                    //     data storage
                    // - note: this change would only involve modifying the
                    //   implementation, we wouldn't actually alter the
                    //   underlying algorithm
                    assert_eq!(tmp_accum_states.n_states(), team_param.n_teams);
                    for offset in 0..team_param.n_members_per_team {
                        let mut tmp_accum_state = tmp_accum_states.get_state_mut(offset);
                        let i_itr = ((i_global + member_id)..(i_global + chunk_len))
                            .step_by(team_param.n_members_per_team);
                        for i in i_itr {
                            self.reducer.consume(
                                &mut tmp_accum_state,
                                &Datum::from_scalar_value(
                                    self.f.call(self.stream.x_array[i]),
                                    self.stream.weights[i],
                                ),
                            );
                        }
                    }
                },
            );
        } else {
            team.calccontribs_combine_apply(
                binned_statepack,
                &self.reducer,
                bin_index,
                &|tmp_accum_states: &mut StatePackViewMut, member_id: usize| {
                    assert_eq!(tmp_accum_states.n_states(), 1);
                    let mut tmp_accum_state = tmp_accum_states.get_state_mut(0);

                    let i_itr = ((i_global + member_id)..(i_global + chunk_len))
                        .step_by(team_param.n_members_per_team);
                    for i in i_itr {
                        self.reducer.consume(
                            &mut tmp_accum_state,
                            &Datum::from_scalar_value(
                                self.f.call(self.stream.x_array[i]),
                                self.stream.weights[i],
                            ),
                        );
                    }
                },
            );
        }
    }
}
