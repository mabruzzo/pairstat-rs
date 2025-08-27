// TODO remove
// do we have to use this to silence warnings?
#![allow(dead_code)]

// the reason this is named mod.rs has to do with some complexities of how
// testing is handled
//
// we are following the advice of the rust book
// https://doc.rust-lang.org/book/ch11-03-test-organization.html#submodules-in-integration-tests

use std::collections::HashMap;

pub type BinnedStatMap = HashMap<&'static str, Vec<f64>>;

// based on numpy!
// https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
//
// I suspect we'll use this a lot! If we may want to define
// a `assert_isclose!` macro to provide a nice error message (or an
// `assert_allclose!` macro to operate upon arrays)
pub fn isclose(actual: f64, ref_val: f64, rtol: f64, atol: f64) -> bool {
    let actual_nan = actual.is_nan();
    let ref_nan = ref_val.is_nan();
    if actual_nan || ref_nan {
        actual_nan && ref_nan
    } else {
        (actual - ref_val).abs() <= (atol + rtol * ref_val.abs())
    }
}

enum ArgDesignation {
    Actual,
    Expected,
}

// I think this should really be a macro
pub fn assert_consistent_results(
    actual: &HashMap<&'static str, Vec<f64>>,
    expected: &HashMap<&'static str, Vec<f64>>,
    rtol_atol_vals: &HashMap<&'static str, [f64; 2]>,
) {
    // it would be nice to handle the following more gracefully
    // (and provide a better error message!)
    let mut key_problem: Option<ArgDesignation> = None;
    if rtol_atol_vals.len() != actual.len() {
        key_problem = Some(ArgDesignation::Actual);
    } else if rtol_atol_vals.len() != expected.len() {
        key_problem = Some(ArgDesignation::Expected);
    } else {
        for k in rtol_atol_vals.keys() {
            if !actual.contains_key(k) {
                key_problem = Some(ArgDesignation::Actual);
                break;
            } else if !expected.contains_key(k) {
                key_problem = Some(ArgDesignation::Expected);
            }
        }
    }

    match key_problem {
        Some(ArgDesignation::Actual) => {
            panic!("`actual` & `rtol_atol_vals` don't have matching keys")
        }
        Some(ArgDesignation::Expected) => {
            panic!("`expected` & `rtol_atol_vals` don't have matching keys")
        }
        None => (),
    }

    for (key, [rtol, atol]) in rtol_atol_vals {
        let len = expected[key].len();
        assert_eq!(
            actual[key].len(),
            len,
            "the lengths of the '{key}' entry in actual and ref are unequal",
        );

        for i in 0..len {
            let actual_val = actual[key][i];
            let ref_val = expected[key][i];
            assert!(
                isclose(actual_val, ref_val, *rtol, *atol),
                "map[\"{key}\"][{i}] values aren't to within rtol={rtol}, atol={atol}\
            \n  actual   = {actual_val}\
            \n  expected = {ref_val}",
            );
        }
    }
}
