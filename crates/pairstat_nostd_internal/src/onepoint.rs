//! This implements a binned onepoint reduction (e.g. a regular binned mean
//! or binned variance).
//!
//! TODO: We probably also want to implement a version of the logic without
//!       binning so that we can have a simple example of using
//!       `team.calccontribs_combine_apply`

use crate::{
    bins::BinEdges,
    misc::segment_idx_bounds,
    parallel::{BinnedDatum, ReductionSpec, StandardTeamParam, Team},
    reducer::{Comp0Mean, Datum},
    state::StatePackViewMut,
};

/// The plan for computing a binned 1-point binned statistic.
///
/// In more detail, this performs a binned reduction for a stream of data.
/// - Each entry in the stream has a `value`, a `weight`, and an `extra_field`
///   value.
/// - the `extra_field` value is used to determine an entry's bin index.
/// - This is a "1-point statistic" in the sense that an entry's `value` and
///   `weight` directly contribute to the statistic (this contrasts with
///   2-point statistics where pairs of points contribute to the statistic).
///
/// Currently, this just implements a binned weighted average calculation for
/// a stream of scalar values.
///
/// # Primary Purpose
/// This primarily exists to serve as a concrete example of how the machinery
/// for driving parallel reductions can be used to perform a (conceptually)
/// simple calculation. This is useful in 2 contexts:
/// 1. It's a pedagocial resource to help illustrating how the machinery
///    actually works.
/// 2. Because calculating a binned mean is conceptually simple, its really
///    easy to implement a simple version of that logic without the
///    parallelism machinery. This makes the type very useful for actually
///    testing parallelism machinery
///
/// # Another Potential Use Case:
/// There are few minor modifications we could make to this type:
/// - make it possible to construct this from [`crate::CartesianBlock`] or
///   [`crate::UnstructuredPoints`] (those types need modifications to be able
///   to optionally track `extra_field`).
/// - Make it possible to compute mean and variance on vector components
///   for a stream of vector values (rather than just a stream of scalars)
///
/// If we make such modifications this could provide useful convenience
/// functionality for studying turbulence. In more detail:
/// - we often need to know the mean velocity field (i.e. we need to make sure
///   that structure functions are computed from a velocity field with a mean
///   of 0)
/// - sometimes we want to know the variance of the velocity field
///
/// While such calculations are simple enough to implement manually, consider
/// the case with a distributed dataset. If a user has already written logic
/// to compute two-point functions for that distributed dataset, it could be
/// extremely useful to reuse that logic for computing the mean and variance
pub struct OnePointBinned<'a, B: BinEdges> {
    /// the common length (shared by `values`, `weights`, and `extra_field`)
    length: usize,

    /// Holds the values that statistics are computed from
    ///
    /// # Note
    /// if we decide to support vectors, then this should be replaced with
    /// `[&'a [f64]; 3]` (as in [`crate::CartesianBlock`])
    values: &'a [f64],

    /// Holds the weight associated with each value
    weights: &'a [f64],

    /// the `i`th element in this array is used to determine the bin (via
    /// `self.bin_edges` of `self.values[i]`)
    extra_field: &'a [f64],

    /// Specifies bin edges
    bin_edges: B,

    // we may want to allow this to hold a generic type in the future
    reducer: Comp0Mean,
}

impl<'a, B: BinEdges> OnePointBinned<'a, B> {
    /// Construct a new [`OnePointBinned`]
    ///
    /// We probably can replace this with 2 alternative constructors:
    /// 1. construct [`OnePointBinned`] from a [`crate::UnstructuredPoints`]
    /// 2. construct [`OnePointBinned`] from a [`crate::CartesianBlock`]
    pub fn new(
        values: &'a [f64],
        weights: &'a [f64],
        extra_field: &'a [f64],
        reducer: Comp0Mean,
        bin_edges: B,
    ) -> Result<Self, &'static str> {
        let length = values.len();
        if values.is_empty() {
            Err("values can't be empty")
        } else if (length != weights.len()) || (length != extra_field.len()) {
            Err("values, weights, and extra_field must have a common length")
        } else {
            Ok(Self {
                length,
                values,
                weights,
                extra_field,
                bin_edges,
                reducer,
            })
        }
    }

    /// internal helper method called by [`Self::add_contributions`] to
    /// construct the [`BinnedDatum`] associated with index `i`
    #[inline]
    fn get_binned_datum_helper(&self, i: usize) -> BinnedDatum {
        let bin_index = match i < self.length {
            true => self.bin_edges.bin_index(self.extra_field[i]),
            false => None,
        };

        if let Some(bin_index) = bin_index {
            let datum = Datum::from_scalar_value(self.values[i], self.weights[i]);
            BinnedDatum { bin_index, datum }
        } else {
            // won't affect reduction since BinnedDatum::Datum::weight is 0
            BinnedDatum::zeroed()
        }
    }
}

impl<'a, B: BinEdges> ReductionSpec for OnePointBinned<'a, B> {
    type ReducerType = Comp0Mean;

    fn get_reducer(&self) -> &Self::ReducerType {
        &self.reducer
    }

    fn n_bins(&self) -> usize {
        self.bin_edges.n_bins()
    }

    fn team_loop_bounds(&self, _: usize, _: &StandardTeamParam) -> (usize, usize) {
        (0, 1) // <- the equivalent of a noop
    }

    const NESTED_REDUCE: bool = false;

    fn add_contributions<T: Team>(
        &self,
        binned_statepack: &mut T::SharedDataHandle<StatePackViewMut>,
        _team_loop_index: usize,
        team: &mut T,
    ) {
        // get team properties
        let team_id = team.team_id();
        let StandardTeamParam {
            n_teams,
            n_members_per_team,
        } = team.standard_team_info();
        debug_assert!(team_id < n_teams);

        // get the range of indices (for self.values) that will be used to
        // update the value of `binned_statepack`. We split the indices as
        // equitably as possible which among the teams
        let length = self.values.len();
        let (start, stop) = segment_idx_bounds(length, team_id, n_teams);

        // now we iterate over all the indices
        for member0_index in (start..stop).step_by(n_members_per_team) {
            // during a given pass through this loop, a callback is passed
            // into `team.collect_pairs_then_apply`.
            //
            // During this call, the team members collaboratively call
            // `team.collect_pairs_then_apply`, and they pass in a callback.
            // That function:
            // 1. invokes the callback to have each member add the BinnedDatum
            //    value that its responsible for to a collection pad
            // 2. gathers together all of the values in the collection pad
            // 3. has 1 team member serially process all of the BinnedDatum
            //    entries in the collection pad (binned_statepack is updated
            //    by each entry)
            //
            // The precise details vary ever so slightly based on the
            // following branch

            if !T::IS_VECTOR_PROCESSOR {
                // this is the standard scenario. In this case, the callback
                // is invoked once for each member

                team.collect_pairs_then_apply(
                    binned_statepack,
                    self.get_reducer(),
                    &|collect_pad: &mut [BinnedDatum], member_id: usize| {
                        debug_assert_eq!(collect_pad.len(), 1); // sanity check!
                        let i = member0_index + member_id;
                        collect_pad[0] = self.get_binned_datum_helper(i);
                    },
                );
            } else {
                // in this case, the members of the team are supposed to
                // correspond to vector lanes
                // -> in this case, the callback is only called one time. The
                //    premise is that we'll leverage vectorization to
                //    parallelize the work
                // -> currently, this has a placeholder implementation right
                //    now that produces the expected result, but doesn't use
                //    vectorization yet
                team.collect_pairs_then_apply(
                    binned_statepack,
                    self.get_reducer(),
                    &|collect_pad: &mut [BinnedDatum], member_id: usize| {
                        debug_assert_eq!(member_id, 0);
                        debug_assert_eq!(collect_pad.len(), n_members_per_team);
                        // to auto-vectorize, we would probably need to:
                        // - pre-generate more elements than n_members_per_team
                        // - make static guarantees about array alignment & len
                        // (it may be easier to use machinery like glam::DVec to
                        // force the vectorizaiton)

                        #[allow(clippy::needless_range_loop)]
                        for lane_id in 0..n_members_per_team {
                            let i = member0_index + lane_id;
                            collect_pad[lane_id] = self.get_binned_datum_helper(i);
                        }
                    },
                );
            }
        }
    }
}
