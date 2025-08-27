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
struct MkWrappedReducerFn(fn(&Config) -> Result<Box<dyn WrappedReducer>, Error>);

/// returns a hashmap, where keys map the names of two-point calculations to
/// functions that construct the appropriate boxed [`WrappedReducer`] trait
/// objects.
fn build_registry() -> HashMap<String, MkWrappedReducerFn> {
    let out: HashMap<String, MkWrappedReducerFn> = HashMap::from([
        // -------------------------------------------------
        // define mappings for correlation function reducers
        // -------------------------------------------------
        (
            "hist_cf".to_owned(),
            MkWrappedReducerFn(|c: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                let Some(ref edges) = c.hist_reducer_bucket else {
                    return Err(Error::bucket_edge_presence(&c.reducer_name, true));
                };
                if let BinEdgeSpec::Regular(edges) = edges {
                    Ok(Box::new(WrappedReducerImpl {
                        reducer: ComponentSumHistogram::from_bin_edges(edges.clone()),
                        pair_op: PairOperation::ElementwiseMultiply,
                    }) as Box<dyn WrappedReducer>)
                } else {
                    todo!()
                    /*
                    Ok(Box::new(WrappedIrregularHist {
                        pair_op: PairOperation::ElementwiseMultiply,
                    }) as Box<dyn WrappedReducer>)
                    */
                }
            }),
        ),
        (
            "2pcf".to_owned(),
            MkWrappedReducerFn(|c: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                if c.hist_reducer_bucket.is_some() {
                    return Err(Error::bucket_edge_presence(&c.reducer_name, false));
                }
                Ok(Box::new(WrappedReducerImpl {
                    reducer: ComponentSumMean::new(),
                    pair_op: PairOperation::ElementwiseMultiply,
                }) as Box<dyn WrappedReducer>)
            }),
        ),
        // -----------------------------------------------
        // define mappings for structure function reducers
        // -----------------------------------------------
        (
            "hist_astro_sf1".to_owned(),
            MkWrappedReducerFn(|c: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                let Some(ref edges) = c.hist_reducer_bucket else {
                    return Err(Error::bucket_edge_presence(&c.reducer_name, true));
                };
                if let BinEdgeSpec::Regular(edges) = edges {
                    Ok(Box::new(WrappedReducerImpl {
                        reducer: EuclideanNormHistogram::from_bin_edges(edges.clone()),
                        pair_op: PairOperation::ElementwiseSub,
                    }) as Box<dyn WrappedReducer>)
                } else {
                    todo!();
                    /*
                    Ok(Box::new(WrappedIrregularHist {
                        pair_op: PairOperation::ElementwiseSub,
                    }) as Box<dyn WrappedReducer>)*/
                }
            }),
        ),
        (
            "astro_sf1".to_owned(),
            MkWrappedReducerFn(|c: &Config| -> Result<Box<dyn WrappedReducer>, Error> {
                if c.hist_reducer_bucket.is_some() {
                    return Err(Error::bucket_edge_presence(&c.reducer_name, false));
                }
                Ok(Box::new(WrappedReducerImpl {
                    reducer: EuclideanNormMean::new(),
                    pair_op: PairOperation::ElementwiseSub,
                }) as Box<dyn WrappedReducer>)
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
static REDUCER_MAKER_REGISTRY: LazyLock<HashMap<String, MkWrappedReducerFn>> =
    LazyLock::new(build_registry);

/// constructs the appropriate [`WrappedReducer`] trait object that
/// corresponds to the specified configuration
pub(crate) fn wrapped_reducer_from_config(
    config: &Config,
) -> Result<Box<dyn WrappedReducer>, Error> {
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

/// A trait that primarily wraps a [`Reducer`] type in order to support
/// dynamic dispatch.
///
/// The main purpose of this function is to help implement the
/// [`crate::AccumulatorDescr`] type.
///
/// # Dynamic Dispatch vs "Enum Dispatch"
///
/// There isn't a particularly strong reason behind our choice of dynamic
/// dispatch. We summarize some considerations down below, but we are very
/// willing to change the implementation.
///
/// Our initial choice was primarily influenced by some concerns about an
/// enum-dispatch approach:
/// - I was originally concerned that enum-dispatch may not work very well
///   with the dynamic plugin approach (for GPU support) that is briefly
///   described in the module-level documentation of [`crate::accumulator`].
///   Upon reflection, I realized that if [`AccumulatorDescr`] continues to
///   treat the [`Config`] type in a similar way to its current treatment
///   (i.e. the [`Config`] instance is stored alongside the [`WrappedReducer`]
///   that was constructed from it and serves as the "source-of-truth" for
///   serialization). **In other words, this isn't a significant concern.**
/// - My main concern focuses on readability and maintainability:
///   - there are "a lot" of Reducer variations to support. Combinatorics from
///     generic types (especially when we want to make different choices for
///     structure functions and correlation functions), significantly increase
///     the number of variations...
///   - Let's consider what it takes to add a new reducer:
///     - under the current system, we just need to update [`build_registry`].
///     - under an enum-system, we would need to update
///       - [`build_registry`] (or whatever equivalent we shift to using).
///       - The enum-system would also have equivalent methods for each method
///         that this trait currently declares (there are 5 at the time of
///         writing), and there would be a match statement over all variants.
///         We would need to update the variant in every method.
///     - while we could potentially use macros to reduce code duplication, I
///       worry about readability
///       - to be fair, we do now use a macro in [`WrappedIrregularHist`].
///         With that said, this covers very special cases (that we may want
///         to consider removing...).
///       - if we don't remove [`WrappedIrregularHist`], we would need to have
///         special handling in the enum-dispatch approach for these special
///         cases (which could significantly complicate the macros)
/// - At a surface-level, it might seem like the enum-system would let us
///   remove all machinery to make [`Config`] implement the [`Eq`] trait
///   (namely, the machinery to implement bin-edge equality checks)
///   - in practice, that isn't really true.
///   - performing equality checks for distance edge bins and Histogram
///     reducers requires us to implement the vast majority of this machinery
///     anyway.
///
/// There are some other considerations:
/// - do we want to support a future where external code can implement new
///   Reducers? (At the moment, we don't)
/// - is the performance benefit of enum-dispatch significant
///   - I'm sketipical, at the moment
///   - But, I could totally see it being more important if we supported
///     multiple reducers per accumulator.
///
/// Importantly, I'm happy to be proven wrong and change our approach. If
/// anybody wants to implement an enum-dispatch approach, I would happily
/// consider it!
pub(crate) trait WrappedReducer {
    /// merge the state information tracked by `binned_statepack` and
    /// `other_binned_statepack`, and update `binned_statepack` accordingly
    ///
    /// # Note
    /// The `config` argument **must** be identical to the value passed
    /// into [`wrapped_reducer_from_config`]. It is _only_ used to help
    /// implement the [`WrappedIrregularHist`] type
    fn merge(
        &self,
        binned_statepack: &mut StatePackViewMut,
        other_binned_statepack: &StatePackView,
        config: &Config,
    );

    /// compute the output quantities from the accumulator's state and return
    /// the result in a HashMap.
    ///
    /// # Note
    /// The `config` argument **must** be identical to the value passed
    /// into [`wrapped_reducer_from_config`]. It is _only_ used to help
    /// implement the [`WrappedIrregularHist`] type
    fn get_output(
        &self,
        binned_statepack: &StatePackView,
        config: &Config,
    ) -> HashMap<&'static str, Vec<f64>>;

    /// Reset the state within the binned statepack.
    ///
    /// # Note
    /// The `config` argument **must** be identical to the value passed
    /// into [`wrapped_reducer_from_config`]. it is _only_ used to help
    /// implement the [`wrappedirregularhist`] type
    fn reset_full_statepack(&self, binned_statepack: &mut StatePackViewMut, config: &Config);

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
    fn accum_state_size(&self, config: &Config) -> usize;

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
    fn exec_reduction(
        &self,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: SpatialInfo,
        config: &Config,
    ) -> Result<(), Error>;
}

/// Represents a wrapped [`Reducer`].
///
/// This type wraps **any** kind of reducer, except for [`ScalarHistogram`]
/// defined using the [`IrregularBinEdges`] type. That case requires special
/// handling (see [`WrappedIrregularHist`] for more details).type for more details.
///  This is a special case among reducers.  will be
/// annoying. We will have to play a "game," where the [`Reducer`] doesn't
/// have persistent state. Instead, we will have to pass in the config to each
/// method and reconstruct the reducer exactly when we need it...
struct WrappedReducerImpl<R: Reducer + Clone> {
    reducer: R,
    pair_op: PairOperation,
}

impl<R: Reducer + Clone> WrappedReducer for WrappedReducerImpl<R> {
    fn merge(
        &self,
        binned_statepack: &mut StatePackViewMut,
        other_binned_statepack: &StatePackView,
        _config: &Config,
    ) {
        merge_full_statepacks(&self.reducer, binned_statepack, other_binned_statepack);
    }

    // we should probably let the caller provide the output memory
    // (it would be nice if we could pass in a type that is more clearly
    // immutable than StatePackViewMut)
    fn get_output(
        &self,
        binned_statepack: &StatePackView,
        _config: &Config,
    ) -> HashMap<&'static str, Vec<f64>> {
        get_output(&self.reducer, binned_statepack)
    }

    fn reset_full_statepack(&self, binned_statepack: &mut StatePackViewMut, _config: &Config) {
        reset_full_statepack(&self.reducer, binned_statepack)
    }

    fn accum_state_size(&self, _config: &Config) -> usize {
        self.reducer.accum_state_size()
    }

    // we probably need to pass other arguments...
    fn exec_reduction(
        &self,
        binned_statepack: &mut StatePackViewMut,
        spatial_info: SpatialInfo,
        config: &Config,
    ) -> Result<(), Error> {
        exec_reduction_helper(
            &self.reducer,
            binned_statepack,
            spatial_info,
            &config.squared_distance_bin_edges,
            self.pair_op,
        )
    }
}

/// Represents a [`ScalarHistogram`] with the [`IrregularBinEdges`] type.
///
/// # Note
/// We have to play a "game," where the [`Reducer`] doesn't have persistent
/// state. Instead, we reconstruct the reducer exactly when we need it.
/// This is done to avoid lifetime issues in [`AccumulatorDescr`]
struct WrappedIrregularHist {
    pair_op: PairOperation,
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

impl WrappedReducer for WrappedReducerNew {
    fn merge(
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

    fn get_output(
        &self,
        binned_statepack: &StatePackView,
        config: &Config,
    ) -> HashMap<&'static str, Vec<f64>> {
        forward_reducer!(
            make_reducer(config); get_output(reducer_ref, binned_statepack)
        )
    }

    fn reset_full_statepack(&self, binned_statepack: &mut StatePackViewMut, config: &Config) {
        forward_reducer!(
            make_reducer(config); reset_full_statepack(reducer_ref, binned_statepack)
        )
    }

    fn accum_state_size(&self, config: &Config) -> usize {
        // this can't be a closure and accept a generic arg
        fn f(reducer: &impl Reducer) -> usize {
            reducer.accum_state_size()
        }

        forward_reducer!(
            make_reducer(config); f(reducer_ref,) // the macro needs trailing `,`
        )
    }

    fn exec_reduction(
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

fn make_reducer(config: &Config) -> ReducerKind {
    todo!("implement me!");
}

enum ReducerKind<'a> {
    TPCFMean(ComponentSumMean),
    TPCFHistRegular(ComponentSumHistogram<RegularBinEdges>),
    TPCFHistIrregular(ComponentSumHistogram<IrregularBinEdges<'a>>),
    AstroSF1Mean(EuclideanNormMean),
    AstroSF1HistRegular(ComponentSumHistogram<RegularBinEdges>),
    AstroSF1HistIrregular(ComponentSumHistogram<IrregularBinEdges<'a>>),
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
