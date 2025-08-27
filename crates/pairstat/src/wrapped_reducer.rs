//! This module introduces the machinery to wrap the various types that
//! implement [`Reducer`] in a uniform manner and enable runtime polymorphism.
//! This machinery is primarily used to implement [`crate::AccumulatorDescr`]
//!
//! Our current approach for doing this involves dynamic dispatch.

use crate::{
    Error,
    apply::{apply_accum, apply_cartesian},
    reducers::{EuclideanNormHistogram, EuclideanNormMean, get_output},
};

use pairstat_nostd_internal::{
    BinEdges, CartesianBlock, CellWidth, ComponentSumHistogram, ComponentSumMean,
    IrregularBinEdges, PairOperation, Reducer, RegularBinEdges, StatePackView, StatePackViewMut,
    UnstructuredPoints, merge_full_statepacks, reset_full_statepack, validate_bin_edges,
};
use std::{collections::HashMap, sync::LazyLock};

/// Wraps a vector holding pre-validated bin edges. This primarily exists so
/// that we can implement the `Eq` trait.
///
/// # Note
/// Ordinarily, [`f64`], and by extension `Vec<f64>` doesn't implement `Eq`
/// since `NaN` != `NaN`. We can implement it here since
/// [`pairstat_nostd_internal::validate_bin_edges`] ensures there aren't any
/// `NaN` values
#[derive(Clone)]
pub(crate) struct ValidatedBinEdgeVec(Vec<f64>);

impl ValidatedBinEdgeVec {
    pub(crate) fn new(edges: Vec<f64>) -> Result<Self, Error> {
        validate_bin_edges(&edges).map_err(Error::internal_legacy_adhoc)?;
        Ok(Self(edges))
    }

    fn as_irregular_edge_view<'a>(&'a self) -> IrregularBinEdges<'a> {
        // TODO consider introducing a way to bypass error checks when
        // we construct IrregularBinEdges from ValidatedBinEdgeVec
        IrregularBinEdges::new(self.0.as_slice()).expect(
            "There must be a bug: either in the pre-validation of the bin \
            edges, OR that somehow mutated bin-edges after they were \
            pre-validated!",
        )
    }
}

impl PartialEq for ValidatedBinEdgeVec {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ValidatedBinEdgeVec {}

/// Holds data that represent bin edges. Importantly, all variants have been
/// pre-validated.
///
/// # Note
/// It's important that this type implements the [`Eq`] trait.
#[derive(Clone, PartialEq, Eq)]
pub(crate) enum BinEdgeSpec {
    Regular(RegularBinEdges),
    Vec(ValidatedBinEdgeVec),
}

impl BinEdgeSpec {
    pub(crate) fn leftmost_edge(&self) -> f64 {
        match self {
            BinEdgeSpec::Regular(edges) => edges.leftmost_edge(),
            BinEdgeSpec::Vec(v) => v.0[0],
        }
    }

    pub(crate) fn n_bins(&self) -> usize {
        match self {
            BinEdgeSpec::Regular(edges) => edges.n_bins(),
            BinEdgeSpec::Vec(v) => v.0.len() - 1,
        }
    }
}

/// A configuration object for a two-point calculation.
///
/// It tracks distance bin-edge information and specifies the reducer
/// properties used in the calculation. The basic premise is that this
/// serves as the "single source of truth" for the calculation properties.
/// We want to make it easy to serialize this information to communicate
/// between processes (or potentially to dynamic plugins that implement
/// GPU functionality that is that is briefly described in the module-level
/// documentation of [`crate::accumulator`].)
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Config {
    reducer_name: String,
    // I'm fairly confident that supporting the Histogram with the
    // [`IrregularBinEdges`] type introduces self-referential struct issues
    // (at a slightly higher level)
    hist_reducer_bucket: Option<BinEdgeSpec>,
    // eventually, we should add an option for variance

    // A compelling case could be made that we should be tracking this in
    // a different struct, and focus the current trait on "just" the
    // properties directly related to the Reducer
    squared_distance_bin_edges: BinEdgeSpec,
}

impl Config {
    // a case could be made that to replace this with a function that
    // constructs both a Config and a boxed WrappedReducer in a single
    // operation so that we make sure Config **always** corresponds to a
    // valid calculation...
    pub(crate) fn new(
        reducer_name: String,
        hist_reducer_bucket: Option<BinEdgeSpec>,
        squared_distance_bin_edges: BinEdgeSpec,
    ) -> Config {
        Config {
            reducer_name,
            hist_reducer_bucket,
            squared_distance_bin_edges,
        }
    }
}

/// an internal type that is used to encode the spatial information
#[derive(Clone)]
pub(crate) enum SpatialInfo<'a> {
    Unstructured {
        points_a: UnstructuredPoints<'a>,
        points_b: Option<UnstructuredPoints<'a>>,
    },
    Cartesian {
        block_a: CartesianBlock<'a>,
        block_b: Option<CartesianBlock<'a>>,
        cell_width: CellWidth,
    },
}

/// Wraps a function pointer that constructs, a boxed [`WrappedReducer`] trait
/// object.
///
/// # Note
/// Make this a type-alias doesn't appear to have the desired effect. I suspect
/// that it doesn't properly coerce a closure to function pointer
struct ReducerKindFnHolder(fn(&Config) -> Result<ReducerKind<'_>, Error>);

/// returns a hashmap from the names of two-point calculations to
/// functions that construct the appropriate boxed [`ReducerKind`] instance
fn build_registry() -> HashMap<String, ReducerKindFnHolder> {
    let out: HashMap<String, ReducerKindFnHolder> = HashMap::from([
        // TODO hist_cf and 2pcf can be combined, with the behavior depending on
        // whether or not config.hist_reducer_bucket is Some.
        // Likewise for hist_astro and astro_sf1

        // -------------------------------------------------
        // define mappings for correlation function reducers
        // -------------------------------------------------
        (
            "hist_cf".to_owned(),
            ReducerKindFnHolder(|c: &Config| -> Result<ReducerKind<'_>, Error> {
                match c.hist_reducer_bucket {
                    None => Err(Error::bucket_edge_presence(&c.reducer_name, true)),
                    Some(BinEdgeSpec::Regular(ref edges)) => Ok(ReducerKind::TPCFHistRegular(
                        ComponentSumHistogram::from_bin_edges(edges.clone()),
                    )),
                    Some(BinEdgeSpec::Vec(ref v)) => Ok(ReducerKind::TPCFHistIrregular(
                        ComponentSumHistogram::from_bin_edges(v.as_irregular_edge_view()),
                    )),
                }
            }),
        ),
        (
            "2pcf".to_owned(),
            ReducerKindFnHolder(|c: &Config| -> Result<ReducerKind<'_>, Error> {
                // there should be no bucket edges since this is not a histogram reducer
                match c.hist_reducer_bucket {
                    Some(_) => Err(Error::bucket_edge_presence(&c.reducer_name, false)),
                    None => Ok(ReducerKind::TPCFMean(ComponentSumMean::new())),
                }
            }),
        ),
        // -----------------------------------------------
        // define mappings for structure function reducers
        // -----------------------------------------------
        (
            "hist_astro_sf1".to_owned(),
            ReducerKindFnHolder(|c: &Config| -> Result<ReducerKind<'_>, Error> {
                match c.hist_reducer_bucket {
                    None => Err(Error::bucket_edge_presence(&c.reducer_name, true)),
                    Some(BinEdgeSpec::Regular(ref edges)) => Ok(ReducerKind::AstroSF1HistRegular(
                        EuclideanNormHistogram::from_bin_edges(edges.clone()),
                    )),
                    Some(BinEdgeSpec::Vec(ref v)) => Ok(ReducerKind::AstroSF1HistIrregular(
                        EuclideanNormHistogram::from_bin_edges(v.as_irregular_edge_view()),
                    )),
                }
            }),
        ),
        (
            "astro_sf1".to_owned(),
            ReducerKindFnHolder(|c: &Config| -> Result<ReducerKind, Error> {
                // there should be no bucket edges since this is not a histogram reducer
                match c.hist_reducer_bucket {
                    Some(_) => Err(Error::bucket_edge_presence(&c.reducer_name, false)),
                    None => Ok(ReducerKind::AstroSF1Mean(EuclideanNormMean::new())),
                }
            }),
        ),
    ]);
    out
}

/// global variable that holds a registry that describes the various kinds of
/// two-point calculations that the crate supports.
///
/// This variable is lazily initialized, in a threadsafe manner, using the
/// [`build_registry`] function. In more detail, it holds a hashmap, where
/// keys map the names of known two-point calculations to functions that
/// construct the appropriate boxed [`WrappedReducer`] trait objects.
///
/// # Note
/// This may not be the optimal way to encode this information.
static REDUCER_MAKER_REGISTRY: LazyLock<HashMap<String, ReducerKindFnHolder>> =
    LazyLock::new(build_registry);

fn make_reducer(config: &Config) -> ReducerKind {
    reducer_kind_from_config(config).unwrap() // this is inefficient...
}

/// constructs the appropriate [`WrappedReducer`] trait object that
/// corresponds to the specified configuration
fn reducer_kind_from_config(config: &Config) -> Result<ReducerKind, Error> {
    let name = &config.reducer_name;
    if let Some(func) = REDUCER_MAKER_REGISTRY.get(name) {
        func.0(config)
    } else {
        Err(Error::reducer_name(
            name.clone(),
            REDUCER_MAKER_REGISTRY.keys().cloned().collect(),
        ))
    }
}

/// constructs the appropriate [`WrappedReducer`] trait object that
/// corresponds to the specified configuration
pub(crate) fn wrapped_reducer_from_config(config: &Config) -> Result<WrappedReducerNew, Error> {
    // confirm we can build a ReducerKind
    let _ = reducer_kind_from_config(config)?;
    Ok(WrappedReducerNew)
}

/// Takes a Calls function (`func`) on:
///   - the reducer expression (this can be refer to a variable holding a
///     reducer or be an expression that returns a reducer)
///   - other arguments (passed as arguments to `forward_reducer`)
///
/// Examples:
/// ```text
/// forward_reducer!(make_reducer(config); func(reducer_ref, other_arg1, other_arg2));
/// # trailing comma is necessary for unary functions!
/// forward_reducer!(make_reducer(config); func(reducer_ref,));
/// ```
macro_rules! forward_reducer{
    ($reducer:expr; $func:ident(reducer_ref, $($args:expr),*)) => {
        {
            match $reducer {
                ReducerKind::TPCFMean(r) => $func(&r, $($args),*),
                ReducerKind::TPCFHistRegular(r) =>  $func(&r, $($args),*),
                ReducerKind::TPCFHistIrregular(r) =>  $func(&r, $($args),*),
                ReducerKind::AstroSF1Mean(r) =>  $func(&r, $($args),*),
                ReducerKind::AstroSF1HistRegular(r) =>  $func(&r, $($args),*),
                ReducerKind::AstroSF1HistIrregular(r) =>  $func(&r, $($args),*),
            }
        }
    }
}

pub(crate) struct WrappedReducerNew;

impl WrappedReducerNew {
    /// merge the state information tracked by `binned_statepack` and
    /// `other_binned_statepack`, and update `binned_statepack` accordingly
    ///
    /// # Note
    /// The `config` argument **must** be identical to the value passed
    /// into [`wrapped_reducer_from_config`]. It is _only_ used to help
    /// implement the [`WrappedIrregularHist`] type
    pub(crate) fn merge(
        &self,
        binned_statepack: &mut StatePackViewMut,
        other_binned_statepack: &StatePackView,
        config: &Config,
    ) {
        forward_reducer!(
            make_reducer(config);
            merge_full_statepacks(reducer_ref, binned_statepack, other_binned_statepack)
        )
    }

    /// compute the output quantities from the accumulator's state and return
    /// the result in a HashMap.
    ///
    /// # Note
    /// The `config` argument **must** be identical to the value passed
    /// into [`wrapped_reducer_from_config`]. It is _only_ used to help
    /// implement the [`WrappedIrregularHist`] type
    pub(crate) fn get_output(
        &self,
        binned_statepack: &StatePackView,
        config: &Config,
    ) -> HashMap<&'static str, Vec<f64>> {
        forward_reducer!(
            make_reducer(config); get_output(reducer_ref, binned_statepack)
        )
    }

    /// Reset the state within the binned statepack.
    ///
    /// # Note
    /// The `config` argument **must** be identical to the value passed
    /// into [`wrapped_reducer_from_config`]. it is _only_ used to help
    /// implement the [`wrappedirregularhist`] type
    pub(crate) fn reset_full_statepack(
        &self,
        binned_statepack: &mut StatePackViewMut,
        config: &Config,
    ) {
        forward_reducer!(
            make_reducer(config); reset_full_statepack(reducer_ref, binned_statepack)
        )
    }

    /// Returns the size of individual accumulator states.
    ///
    /// In a binned_statepack, the total number of entries is the product of
    /// the number returned by this method and the number of bins.
    ///
    /// # Notes
    /// While the number of outputs per bin is commonly the same as the value
    /// returned by this function, that need not be the case. For example,
    /// imagine we used an algorithm like Kahan summation to attain improved
    /// accuracy.
    ///
    /// The `config` argument **must** be identical to the value passed
    /// into [`wrapped_reducer_from_config`]. It is _only_ used to help
    /// implement the [`WrappedIrregularHist`] type
    pub(crate) fn accum_state_size(&self, config: &Config) -> usize {
        // this can't be a closure and accept a generic arg
        fn f(reducer: &impl Reducer) -> usize {
            reducer.accum_state_size()
        }

        forward_reducer!(
            make_reducer(config); f(reducer_ref,) // the macro needs trailing `,`
        )
    }

    /// Executes the reduction on the supplied spatial data and updates
    /// binned_statepack accordingly.
    ///
    /// # Extra Arguments
    /// We will definitely need to accept more arguments in the future
    /// (e.g. to control parallelism)
    ///
    /// # Note
    /// The `config` argument **must** be identical to the value passed
    /// into [`wrapped_reducer_from_config`]. We _only_ require a full
    /// [`Config`] instance, rather than _just_ the distance bin edges in
    /// order to help implement the [`WrappedIrregularHist`] type
    ///
    /// # Why this exists?
    /// At a high-level, one might ask why does this exist? The full reduction
    /// has multiple inputs and it doesn't really seem like it should be a
    /// method of the Reducer... The answer: "because it has to."
    ///
    /// Aside: In an enum-dispatch approach, the reduction-launcher function
    /// wouldn't *need* to directly be a method attached to the enum, but the
    /// function would still need to have access to all the enum's details
    /// (in order to perform a match over every variant)
    pub(crate) fn exec_reduction(
        &self,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: SpatialInfo,
        config: &Config,
    ) -> Result<(), Error> {
        let reducer = make_reducer(config);
        let pair_op = reducer.pair_op();

        forward_reducer!(
            reducer;
            exec_reduction_helper(
                reducer_ref,
                binned_statepack,
                spatial_info,
                &config.squared_distance_bin_edges,
                pair_op
            )
        )
    }
}

enum ReducerKind<'a> {
    TPCFMean(ComponentSumMean),
    TPCFHistRegular(ComponentSumHistogram<RegularBinEdges>),
    TPCFHistIrregular(ComponentSumHistogram<IrregularBinEdges<'a>>),
    AstroSF1Mean(EuclideanNormMean),
    AstroSF1HistRegular(EuclideanNormHistogram<RegularBinEdges>),
    AstroSF1HistIrregular(EuclideanNormHistogram<IrregularBinEdges<'a>>),
}

impl<'a> ReducerKind<'a> {
    fn pair_op(&self) -> PairOperation {
        match self {
            Self::TPCFMean(_) => PairOperation::ElementwiseMultiply,
            Self::TPCFHistRegular(_) => PairOperation::ElementwiseMultiply,
            Self::TPCFHistIrregular(_) => PairOperation::ElementwiseMultiply,
            Self::AstroSF1Mean(_) => PairOperation::ElementwiseSub,
            Self::AstroSF1HistRegular(_) => PairOperation::ElementwiseSub,
            Self::AstroSF1HistIrregular(_) => PairOperation::ElementwiseSub,
        }
    }
}

fn exec_reduction_helper<R: Clone + Reducer>(
    reducer: &R,
    binned_statepack: &mut StatePackViewMut,
    spatial_info: SpatialInfo,
    squared_distance_bin_edges: &BinEdgeSpec,
    pair_op: PairOperation,
) -> Result<(), Error> {
    // this can't be a closure if it accepts generic parameters
    fn inner<R: Clone + Reducer, B: BinEdges + Clone>(
        reducer: &R,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: &SpatialInfo,
        squared_distance_bin_edges: &B,
        pair_op: PairOperation,
    ) -> Result<(), Error> {
        match spatial_info {
            SpatialInfo::Unstructured { points_a, points_b } => apply_accum(
                binned_statepack,
                reducer,
                points_a,
                points_b.as_ref(),
                squared_distance_bin_edges,
                pair_op,
            ),
            SpatialInfo::Cartesian {
                block_a,
                block_b,
                cell_width,
            } => apply_cartesian(
                binned_statepack,
                reducer,
                block_a,
                block_b.as_ref(),
                cell_width,
                squared_distance_bin_edges,
                pair_op,
            ),
        }
    }

    match squared_distance_bin_edges {
        BinEdgeSpec::Vec(v) => inner(
            reducer,
            binned_statepack,
            &spatial_info,
            &v.as_irregular_edge_view(),
            pair_op,
        ),
        BinEdgeSpec::Regular(edges) => {
            inner(reducer, binned_statepack, &spatial_info, edges, pair_op)
        }
    }
}
