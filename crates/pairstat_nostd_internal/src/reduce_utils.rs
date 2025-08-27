// this defines some basic utilities used in reductions.
// it's unclear if we want these things to be part of the public API.
// But, in the short term, as we start to implement things, these are quite
// useful utilities

use crate::reducer::Reducer;
use crate::state::{StatePackView, StatePackViewMut};

// not sure if this should actually be part of the public API, but it's useful
// in a handful of cases
pub fn reset_full_statepack(reducer: &impl Reducer, statepack: &mut StatePackViewMut) {
    for i in 0..statepack.n_states() {
        reducer.init_accum_state(&mut statepack.get_state_mut(i));
    }
}

// ideally, other would be more clearly immutable, but I don't think we want to
// introduce another type just for this 1 case
//
// not sure if this should actually be part of the public API, but it's useful
// in a handful of cases
pub fn merge_full_statepacks(
    reducer: &impl Reducer,
    statepack: &mut StatePackViewMut,
    other: &StatePackView,
) {
    let n_bins = statepack.n_states();
    assert_eq!(n_bins, other.n_states());
    for i in 0..statepack.n_states() {
        reducer.merge(&mut statepack.get_state_mut(i), &other.get_state(i));
    }
}

// consolidates the statepacks in such a way that scratch_statepacks[0]
// contains the results of every other statepack
//
// this function makes no guarantees about the final state of other
// entries within scratch_statepack
//
// not sure if this should actually be part of the public API, but it's useful
// in a handful of cases
pub fn serial_consolidate_scratch_statepacks(
    reducer: &impl Reducer,
    scratch_statepacks: &mut [StatePackViewMut],
) {
    // if we wanted to simulate parallelism, then the way this loop works is
    // pretty inefficient. We would instead strive for something more similar
    // to serial_merge_accum_states
    for i in 1..scratch_statepacks.len() {
        let [main, other] = scratch_statepacks.get_disjoint_mut([0, i]).unwrap();
        merge_full_statepacks(reducer, main, &other.as_view());
    }
}

// Merges all accumulator states. At the end of this function call,
// statepack.get_state(0) is valid. All other entries are in an undetermined
// state
//
// not sure if this should actually be part of the public API, but it's useful
// in a handful of cases
#[allow(dead_code)]
pub fn serial_merge_accum_states(_reducer: &impl Reducer, _statepack: &mut StatePackViewMut) {
    todo!("this needs to be implemented eventually. We probably need to rethink things");
    /*
    let n_states = statepack.n_states();

    // we currently need to access the underlying array view in order to safely
    // get disjoint accum_states (where one of them is mutable). When we get
    // drop ndarray, I suspect that this logic may need to be unsafe
    let mut statepack_buf = statepack.as_array_view_mut();

    if !n_states.is_power_of_two() {
        // `1` is considered a power of 2
        todo!("haven't implemented reduction logic in this scenario!")
    }
    let mut remaining_pairs = n_states / 2;

    while remaining_pairs > 0 {
        for i in 0..remaining_pairs {
            let (left_buf, right_buf) =
                statepack_buf.multi_slice_mut((s![.., i], s![.., i + remaining_pairs]));

            reducer.merge(
                &mut AccumStateViewMut::from_array_view(left_buf),
                &AccumStateViewMut::from_array_view(right_buf).as_view(),
            );
        }
        remaining_pairs /= 2;
    }
    */
}
