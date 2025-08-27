use ndarray::ArrayView2;
use pairstat::{Comp0Histogram, Comp0Mean, Reducer, get_output_from_statepack_array};
use pairstat_nostd_internal::{AccumStateView, AccumStateViewMut, Datum};
use std::collections::HashMap;

// this is inefficient, but it gets the job done for now
//
// this is probably an indication that we could improve the Reducer/accumulation
// API
fn _get_output_single(
    reducer: &impl Reducer,
    accum_state: &AccumStateView,
) -> HashMap<&'static str, Vec<f64>> {
    let len = accum_state.len();
    assert!(len > 0, "accum_state should never be empty");
    let buf = Vec::<f64>::from_iter((0..len).map(|i| accum_state[i]));
    let view = ArrayView2::from_shape([len, 1], &buf).unwrap();
    get_output_from_statepack_array(reducer, &view)
}

// TODO: factor out this function and the get_output function from
//       integration_tests.rs (they are identical)

#[cfg(test)]
mod tests {

    use super::*;
    use pairstat_nostd_internal::RegularBinEdges;

    #[test]
    fn mean_consume_once() {
        let reducer = Comp0Mean::new();

        let mut storage = [0.0, 0.0];
        let mut accum_state = AccumStateViewMut::from_contiguous_slice(&mut storage);
        reducer.init_accum_state(&mut accum_state);

        reducer.consume(&mut accum_state, &Datum::from_scalar_value(4.0, 1.0));
        let value_map = _get_output_single(&reducer, &accum_state.as_view());

        assert_eq!(value_map["mean"][0], 4.0);
        assert_eq!(value_map["weight"][0], 1.0);
    }

    #[test]
    fn mean_consume_twice() {
        let reducer = Comp0Mean::new();

        let mut storage = [0.0, 0.0];
        let mut accum_state = AccumStateViewMut::from_contiguous_slice(&mut storage);
        reducer.init_accum_state(&mut accum_state);

        reducer.consume(&mut accum_state, &Datum::from_scalar_value(4.0, 1.0));
        reducer.consume(&mut accum_state, &Datum::from_scalar_value(8.0, 1.0));

        let value_map = _get_output_single(&reducer, &accum_state.as_view());
        assert_eq!(value_map["mean"][0], 6.0);
        assert_eq!(value_map["weight"][0], 2.0);
    }

    #[test]
    fn merge() {
        let reducer = Comp0Mean::new();

        let mut storage = [0.0, 0.0];
        let mut accum_state = AccumStateViewMut::from_contiguous_slice(&mut storage);
        reducer.init_accum_state(&mut accum_state);
        reducer.consume(&mut accum_state, &Datum::from_scalar_value(4.0, 1.0));
        reducer.consume(&mut accum_state, &Datum::from_scalar_value(8.0, 1.0));

        let mut storage_other = [0.0, 0.0];
        let mut accum_state_other = AccumStateViewMut::from_contiguous_slice(&mut storage_other);
        reducer.init_accum_state(&mut accum_state_other);
        reducer.consume(&mut accum_state_other, &Datum::from_scalar_value(1.0, 1.0));
        reducer.consume(&mut accum_state_other, &Datum::from_scalar_value(3.0, 1.0));

        reducer.merge(&mut accum_state, &accum_state_other.as_view());

        let value_map = _get_output_single(&reducer, &accum_state.as_view());
        assert_eq!(value_map["mean"][0], 4.0);
        assert_eq!(value_map["weight"][0], 4.0);
    }

    #[test]
    fn hist_consume() {
        let reducer = Comp0Histogram::from_bin_edges(RegularBinEdges::new(0.0, 2.0, 2).unwrap());

        let mut storage = [0.0, 0.0];
        let mut accum_state = AccumStateViewMut::from_contiguous_slice(&mut storage);
        reducer.init_accum_state(&mut accum_state);

        reducer.consume(&mut accum_state, &Datum::from_scalar_value(0.5, 1.0));
        reducer.consume(&mut accum_state, &Datum::from_scalar_value(-50.0, 1.0));
        reducer.consume(&mut accum_state, &Datum::from_scalar_value(1000.0, 1.0));

        let value_map = _get_output_single(&reducer, &accum_state.as_view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 0.0);

        reducer.consume(&mut accum_state, &Datum::from_scalar_value(1.1, 5.0));
        let value_map = _get_output_single(&reducer, &accum_state.as_view());
        assert_eq!(value_map["weight"][0], 1.0);
        assert_eq!(value_map["weight"][1], 5.0);
    }
}
