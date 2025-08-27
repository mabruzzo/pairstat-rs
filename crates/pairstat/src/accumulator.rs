//! This implements a quick-and-dirty first attempt at the "high-level" API
//!
//! # A bird's eye view (motivation)
//! - we leverage Rust's generics and type system to write a lot of highly
//!   specialized code paths
//! - however, there is a large combinatorial problem with selecting the
//!   appropriate configuration.
//!   - we don't want to demand that users be familiar with all of the
//!     highly tuned implementation paths. A similar philosophy applies to
//!     other crates and libraries (e.g. the regex crate)
//!   - we want a more dynamical interface, where you can treat different
//!     variations of a calculation in a very similar manner
//! - furthermore, we need to make all of the variants easy to dynamically
//!   access to expose them to a python library.
//!
//! # Guiding Principles:
//! - It's clear that we are going to use an accumulator-oriented API
//! - We need to be mindful of 2 important use-cases: (i) the fact that we
//!   want people to be able easily build MPI programs that use this library
//!   as a building block and (ii) the fact that people may be computing
//!   multiple statistics. This means that
//!   1. our accumulator API must be easy to serialize (for communicating
//!      between machines)
//!      - for speed purposes, we probably want to serialize accumulator
//!        config & accumulator state separately. We might do the former
//!        with JSON. The latter holds more information, but can probably
//!        be treated as a stream of bytes
//!   2. we need to think about "equality checking"  and user expectations:
//!      - if we have a single monolithic accumulator object, containing state
//!        & configuration, the users would probably expect that we'll check
//!        equality to prevent the merging of unrelated errors.
//!      - if we store the state and configuration separately, I think it
//!        becomes much more obvious that keeping track of this is the user's
//!        responsibility
//!      - it turns out that equality checking can be tricky!
//!
//! - GPU support consideration. This is a very low priority, but we should
//!   pay some attention to what this may entail... In other words, we don't
//!   want to throw everything away. In more detail"
//!   - full GPU support requires a lot of thought. There are lots of things
//!     that we *could* choose to support.
//!   - in my mind, to ship experimental GPU support (especially if we want
//!     to provide it in a python package on PyPI) we probably want to put the
//!     GPU accelerated functionality in a shared library they we treat as a
//!     Run-Time Loadable Extension/Plugin.
//!     - Basically, we'd try to use `dlopen` to bring in functions (see the
//!       libloading crate). This would let us conditionally provide the
//!       functionality **if the hardware and dependencies are present on the
//!       system**
//!     - this could also act as a mechanism for optionally providing SIMD
//!       accelerated functions
//!   - doing this wouldn't be terribly difficult
//!
//!
//! # Other Thoughts
//! I fully expect that this initial API will be suboptimal. Personal
//! experienc with the Grackle project has clearly established that API design
//! is very hard! To add to this:
//! - I have minimal experience designing APIs for languages with
//!   sophisticated type-systems like C++/Rust
//! - the fact that I'm focusing so much on binding to python, and considering
//!   other constrains probably means that the design may be a little
//!   suboptimal for rust.
//!
//! Finally, there are 2 topics that I'm intentionally ignoring in this first
//! implementation. There are hugely important, but I can't keep track of
//! everything and need to start writing code! I am hoping that choices become
//! more apparent later on:
//! 1. How do we handle data-ownership and minimize unnecessary copies? This
//!    initial implementation copies arrays, particularly in the context of
//!    bin edges, very aggressively!
//!    - for context, the underlying algorithms are designed to operate on
//!      slices.
//!    - this has important implications for serialization
//!    - I'm being aggressive to simplify things, because the actual
//!      calculation can be **VERY** expensive "at scale" (i.e. a little extra
//!      overhead won't affect initial library adoption)
//! 2. relatedly, should we be organizing the API within some overarching
//!    context that manages memory? This is a very C oriented way of thinking
//!    (basically all "objects" a user interacts with would be handles). My
//!    gut instinct is "no" (we can get by with "smart pointers"), but it
//!    would certainly simplify some things...
use crate::{
    Error,
    wrapped_reducer::{
        BinEdgeSpec, Config, SpatialInfo, ValidatedBinEdgeVec, WrappedReducer, WrappedReducerNew,
        wrapped_reducer_from_config,
    },
};
use std::collections::HashMap;

use pairstat_nostd_internal::{RegularBinEdges, StatePackView, StatePackViewMut};

// current design:
// -> we have an `Accumulator`, which is composed of an `AccumulatorDescr` &
//    an `AccumulatorData`
// -> `AccumulatorDescr` tracks distance bin edges, the Reducer, and the
//    `PairOperation` instance
//    -> I think this may be a little too "coarse."
//       -> I always liked the idea of associating multiple accumulators with
//          a single set of distance bin edges
//       -> there is more stuff to validate when we call `Accumulator::merge``
//       -> but, we've got to start *somewhere*
//    -> To start, we are going to store a copy of the configuration
//       within `AccumulatorDescr` and use it for equality-checks and for
//       serialization (it's suboptimal, but it simplifies a lot about the
//       initial implementation)

/// Represents an accumulator
///
/// # Note
/// I remain somewhat unconvinced that we really want this type to exist. It
/// may be better to just have people use the constituent parts
///
/// # Implementation Note
/// This should probably not implement [`Clone`]. See [`AccumulatorData`] for
/// more details
pub struct Accumulator {
    data: AccumulatorData,
    descr: AccumulatorDescr,
}

impl Accumulator {
    /// merge the state information tracked by `accum_state` and `other`, and
    /// update `accum_state` accordingly
    pub fn merge(&mut self, other: &Accumulator) -> Result<(), Error> {
        if self.descr == other.descr {
            self.descr.reducer.merge(
                &mut self.data.mut_view(),
                &other.data.view(),
                &self.descr.config,
            );
            Ok(())
        } else {
            todo!("create and return the appropriate error type");
        }
    }

    /// compute the output quantities from the accumulator's state and return
    /// the result in a HashMap.
    ///
    /// # Note
    /// We should probably let the caller provide the output memory.
    pub fn get_output(&self) -> HashMap<&'static str, Vec<f64>> {
        self.descr
            .reducer
            .get_output(&self.data.view(), &self.descr.config)
    }

    /// Reset the tracked accumulator state
    pub fn reset_data(&mut self) {
        self.descr
            .reducer
            .reset_full_statepack(&mut self.data.mut_view(), &self.descr.config);
    }
}

// this is the logic that actually executes the reductions
impl Accumulator {
    /// Executes the reduction on the supplied spatial data and updates
    /// binned_statepack accordingly.
    ///
    /// # Extra Arguments
    /// We will definitely need to accept more arguments in the future
    /// (e.g. to control parallelism)
    ///
    /// # Why this exists?
    /// At a high-level, one might ask why does this exist? The full reduction
    /// has multiple inputs and it doesn't really seem like it should be a
    /// method of the accumulator... The short answer: "because it has to."
    ///
    /// In slightly more detail, the full reduction uses generics to
    /// specialize (for optimization purposes) the calculation. Recall that
    /// this effectively "erases" the reducer types by wrapping them in
    /// trait objects. To call the full reduction we need to "unerase" the
    /// reducer; the only way to do that is in a method of the trait object.
    pub(crate) fn exec_reduction<'a>(
        &mut self,
        spatial_info: SpatialInfo<'a>,
    ) -> Result<(), Error> {
        self.descr.reducer.exec_reduction(
            &mut self.data.mut_view(),
            spatial_info,
            &self.descr.config,
        )
    }
}

/// Holds the data associated with an accumulator
///
/// # Implementation Note
/// It would be **really** nice for this type to be able to wrap externally
/// allocated data. In my mind there are 3 ways to do this:
/// 1. Hold an enum where one variant holds a [`Vec`] and the other variant
///    holds a mutable slice. In this case we would need to associate a
///    lifetime parameter with the type
/// 2. Hold an enum where one variant holds a [`Vec`] and the other variant
///    tracks `Arc<RefCell<T>>`, where `T` is a custom type that can be
///    converted to a slice
/// 3. Some kind of registry approach (we would need to associate a lifetime
///    parameter with this type)
/// 4. We could have some kind of unsafe api where we pass in a pointer and
///    have this type take ownership (in addition to the normal vector
///    representation)
///
/// The downside of introducing a lifetime parameter is that its "infectious".
/// The [`Accumulator`] will also need to have a lifetime parameter, since it
/// holds this type.
///
/// For now, we explicitly choose not to derive [`Clone`] since some of these
/// choices are somewhat incompatible with cloning.
struct AccumulatorData {
    // it would be nice to hold externally allocated data...
    // -> Maybe the answer is to create an enum with 2 variants?
    //    - the 1st variant is `Vec<f64>`
    //    - the other variant is
    data: Vec<f64>,
    n_state: usize,
    state_size: usize,
}

impl AccumulatorData {
    // I'm hesitant to implement DerefMut because it forces us to publicly
    // expose StatePackViewMut
    fn mut_view<'a>(&'a mut self) -> StatePackViewMut<'a> {
        StatePackViewMut::from_slice(self.n_state, self.state_size, self.data.as_mut_slice())
    }

    fn view<'a>(&'a self) -> StatePackView<'a> {
        StatePackView::from_slice(self.n_state, self.state_size, self.data.as_slice())
    }
}

/// Encapsulates the accumulator properties
struct AccumulatorDescr {
    config: Config,
    reducer: WrappedReducerNew,
}

impl Clone for AccumulatorDescr {
    fn clone(&self) -> AccumulatorDescr {
        AccumulatorDescr {
            config: self.config.clone(),
            // since self was already constructed from self.config, we should
            // be able to construct the new instance from config
            reducer: wrapped_reducer_from_config(&self.config).expect("there is a bug"),
        }
    }
}

impl PartialEq for AccumulatorDescr {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
    }
}

impl Eq for AccumulatorDescr {}

enum RawBinEdgeSpec {
    EdgeVec(Vec<f64>),  // unvalidated
    Edge2Vec(Vec<f64>), // unvalidated
    Regular(RegularBinEdges),
    Regular2(RegularBinEdges),
}

#[derive(Default)]
pub struct AccumulatorBuilder {
    // ugh... its very inelegant that we aren't directly modifying a Config
    // object! But, I think it's necessary given that Config isn't meant to be
    // partially initialized
    calc_kind: Option<String>,
    bucket_edges: Option<RawBinEdgeSpec>,
    distance_bin_edges: Option<RawBinEdgeSpec>,
}

impl AccumulatorBuilder {
    // unclear whether we should pass any args to new. It doesn't seem to be
    // the general practice.
    pub fn new() -> AccumulatorBuilder {
        AccumulatorBuilder::default()
    }

    /// Specify the kind of calculation
    ///
    /// All error-handling occurs during [`AccumulatorBuilder::build`]
    ///
    /// # Developer Note
    /// Is this really the best way? There is a tension here. Do we use a
    /// curated set of high-level names? In a more feature complete version we
    /// might accept arguments like:
    /// - "2pcf", "hist_cf"
    /// - "hist_astro_sf1", "astro_sf1", "astro_sf2", "sf2_tensor",
    ///   "sf3_tensor"
    /// - "longitudinal_sf2", "longitudinal_sf3"
    ///
    /// Also need to think about the fact that we probably want to be able to
    /// compute variance in each case. That can probably be a separate argument
    pub fn calc_kind(&mut self, calc_kind: &str) -> &mut AccumulatorBuilder {
        self.calc_kind = Some(calc_kind.to_owned());
        self
    }

    // todo: maybe we force people to specify a BinEdgeSpec instead of having
    // 2 versions of this method?
    // - I worry that won't be very ergonomic...
    // - I don't think we can implement the `From` or `Into` traits to convert
    //   an arbitrary slice to BinEdgeSpec since the conversion shouldn't be
    //   allowed to fail. We could only do that with RawBinEdgeSpec, and I
    //   don't want to expose that underlying type...
    pub fn hist_bucket_edges(&mut self, edges: &[f64]) -> &mut AccumulatorBuilder {
        self.bucket_edges = Some(RawBinEdgeSpec::EdgeVec(edges.to_vec()));
        self
    }

    pub fn regular_hist_bucket_edges(&mut self, edges: RegularBinEdges) -> &mut AccumulatorBuilder {
        self.bucket_edges = Some(RawBinEdgeSpec::Regular(edges.clone()));
        self
    }

    /// Specifies the squared distance bin edges for the calculation
    ///
    /// This overwrites any previously specified value (including values
    /// specified with [`Self::dist_bin_edges`] or
    /// [`Self::regular_dist2_bin_edges`])
    pub fn dist2_bin_edges(&mut self, edges: &[f64]) -> &mut AccumulatorBuilder {
        // do we really need this method?  (I think we can probably delete it...)
        self.distance_bin_edges = Some(RawBinEdgeSpec::Edge2Vec(edges.to_vec()));
        self
    }

    /// Specifies the squared distance bin edges for the calculation
    ///
    /// This overwrites any previously specified value (including values
    /// specified with [`Self::dist_bin_edges`] or [`Self::dist2_bin_edges`])
    pub fn regular_dist2_bin_edges(&mut self, edges: RegularBinEdges) -> &mut AccumulatorBuilder {
        // do we really want this method?
        self.distance_bin_edges = Some(RawBinEdgeSpec::Regular2(edges));
        self
    }

    /// Specifies the distance bin edges for the calculation
    ///
    /// This overwrites any previously specified value (including values
    /// specified with [`Self::dist_bin_edges`] or
    /// [`Self::regular_dist2_bin_edges`])
    pub fn dist_bin_edges(&mut self, edges: &[f64]) -> &mut AccumulatorBuilder {
        self.distance_bin_edges = Some(RawBinEdgeSpec::EdgeVec(Vec::from(edges)));
        self
    }

    pub fn build(&self) -> Result<Accumulator, Error> {
        let hist_reducer_bucket = match &self.bucket_edges {
            Some(RawBinEdgeSpec::EdgeVec(v)) => Some(BinEdgeSpec::Vec(
                ValidatedBinEdgeVec::new(v.clone())
                    .map_err(|err| Error::bin_edge("hist_bucket_edges".to_owned(), err))?,
            )),
            Some(RawBinEdgeSpec::Regular(edges)) => Some(BinEdgeSpec::Regular(edges.clone())),
            Some(_) => panic!("should be unreachable"),
            None => None,
        };

        let squared_distance_bin_edges = match &self.distance_bin_edges {
            Some(RawBinEdgeSpec::EdgeVec(v)) => {
                if v.iter().any(|x| *x < 0.0) {
                    return Err(Error::bin_edge_custom(
                        "distance_bin_edges",
                        "contains negative value",
                    ));
                }
                BinEdgeSpec::Vec(
                    ValidatedBinEdgeVec::new(v.iter().map(|x| x * x).collect()).map_err(|err| {
                        Error::bin_edge("distance_squared_bin_edges".to_owned(), err)
                    })?,
                )
            }
            Some(RawBinEdgeSpec::Edge2Vec(v)) => BinEdgeSpec::Vec(
                ValidatedBinEdgeVec::new(v.clone())
                    .map_err(|err| Error::bin_edge("distance_squared_bin_edges".to_owned(), err))?,
            ),
            Some(RawBinEdgeSpec::Regular(_)) => panic!("should be unreachable!"),
            Some(RawBinEdgeSpec::Regular2(edges)) => BinEdgeSpec::Regular(edges.clone()),
            None => return Err(Error::distance_edge_presence()),
        };

        if squared_distance_bin_edges.leftmost_edge() < 0.0 {
            return Err(Error::bin_edge_custom(
                "distance_squared_bin_edges",
                "contains negative value",
            ));
        }

        // it's more convenient to get the number of states, before we create
        // the config instance
        let n_state = squared_distance_bin_edges.n_bins();

        // construct the Config
        let config = Config::new(
            self.calc_kind.clone().unwrap(),
            hist_reducer_bucket,
            squared_distance_bin_edges,
        );

        let reducer = wrapped_reducer_from_config(&config)?;
        let descr = AccumulatorDescr { config, reducer };

        let state_size = descr.reducer.accum_state_size(&descr.config) as usize;
        let data = AccumulatorData {
            data: vec![0.0; state_size * n_state],
            n_state,
            state_size,
        };
        let mut out = Accumulator { data, descr };
        out.reset_data();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_is_err_contains<T, E: std::error::Error>(rslt: &Result<T, E>, text: &str) {
        let Err(e) = rslt else {
            panic!("result is not an error");
        };
        let err_string = e.to_string();
        if !err_string.contains(text) {
            panic!(
                "error doesn't expand to message containing \"{text}\". The \
                message reads:\n> {err_string}",
            );
        }
    }

    #[test]
    fn valid_name() {
        let rslt = AccumulatorBuilder::new()
            .calc_kind("2pcf")
            .dist_bin_edges(&[0.0, 1.0])
            .build();
        assert!(rslt.is_ok());
    }

    #[test]
    fn valid_hist() {
        let rslt = AccumulatorBuilder::new()
            .calc_kind("hist_cf")
            .dist_bin_edges(&[0.0, 1.0])
            .hist_bucket_edges(&[-5.0, 0.0, 5.0, 10.0])
            .build();
        assert!(rslt.is_ok());
    }

    #[test]
    fn builder_invalid_name() {
        let rslt = AccumulatorBuilder::new()
            .calc_kind("unknown-string")
            .dist_bin_edges(&[0.0, 1.0])
            .build();
        assert_is_err_contains(&rslt, "not a reducer name");
    }

    #[test]
    fn builder_negative_dist_bin_edge() {
        let rslt = AccumulatorBuilder::new()
            .calc_kind("astro_sf1")
            .dist_bin_edges(&[-1.0, 1.0])
            .build();
        assert_is_err_contains(&rslt, "contains negative value");
    }

    #[test]
    fn builder_negative_dist2_bin_edge() {
        let rslt = AccumulatorBuilder::new()
            .calc_kind("astro_sf1")
            .dist2_bin_edges(&[-1.0, 1.0])
            .build();
        assert_is_err_contains(&rslt, "contains negative value");
    }

    #[test]
    fn builder_negative_regular_dist2_bin_edge() {
        let rslt = AccumulatorBuilder::new()
            .calc_kind("astro_sf1")
            .regular_dist2_bin_edges(RegularBinEdges::new(-3.0, 3.0, 4).unwrap())
            .build();
        assert_is_err_contains(&rslt, "contains negative value");
    }

    #[test]
    fn shouldnt_receive_bucket_edges() {
        let rslt = AccumulatorBuilder::new()
            .calc_kind("astro_sf1")
            .dist2_bin_edges(&[0.0, 1.0])
            .regular_hist_bucket_edges(RegularBinEdges::new(-3.0, 3.0, 4).unwrap())
            .build();
        assert_is_err_contains(&rslt, "shouldn't receive bucket edges");

        let rslt = AccumulatorBuilder::new()
            .calc_kind("2pcf")
            .dist2_bin_edges(&[0.0, 1.0])
            .hist_bucket_edges(&[-10.0, 10.0])
            .build();
        assert_is_err_contains(&rslt, "shouldn't receive bucket edges");
    }

    #[test]
    fn didnt_receive_bucket_edges() {
        let rslt = AccumulatorBuilder::new()
            .calc_kind("hist_astro_sf1")
            .dist2_bin_edges(&[0.0, 1.0])
            .build();
        assert_is_err_contains(&rslt, "didn't receive bucket edges");

        let rslt = AccumulatorBuilder::new()
            .calc_kind("hist_cf")
            .dist2_bin_edges(&[0.0, 1.0])
            .build();
        assert_is_err_contains(&rslt, "didn't receive bucket edges");
    }
}
