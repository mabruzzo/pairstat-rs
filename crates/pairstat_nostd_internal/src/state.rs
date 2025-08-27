//! Introduces the [`AccumStateView`] and [`AccumStateViewMut`] types
//!
//! # Why do we need separate types to represent immutable & mutable views?
//!
//! To be more clear: `ndarray::ArrayView1<f64>` and
//! `ndarray::ArrayViewMut1<f64>` seem like direct analogues to the
//! immutable and mutable slice types, `&[f64]` and `&mut [f64]`.
//! **Is there anything fundamentally special about rust's slice types?**
//!
//! It turns out that the answer is "yes"
//! - suppose that you had a type called `StatePackViewMut` that holds a block
//!   of memory and that represents a list of  accumulator states. To use a
//!   type called `AccumStateView` for both mutable and immutable views, you
//!   would have to be write methods with the following signatures:
//!     ```text
//!     impl StatePackViewMut {
//!         fn get_state_view(&self, usize i) -> &AccumStateView;
//!         fn get_state_view_mut(&mut self, usize i) -> &mut AccumStateView;
//!     }
//!     ```
//!   The fact that we return an immutable reference in one case and a mutable
//!   reference in the other case is the key takeaway here. Doing anything else
//!   (with a single view type) won't properly model lifetimes.
//! - This is very allowed if the `StatePackViewMut` already internally tracks
//!   a list of `AccumStateView` instances. But, we would like to avoid doing
//!   that for performance purposes (it would also be messy from a bookkeeping
//!   perspective).
//! - Instead, we would prefer to construct an instance of the `AccumStateView`
//!   return and return a mutable/immutable reference to that instance.
//! - This is forbidden in just about all cases because you would be returning
//!   a dangling reference to a cleaned up local variable.
//! - However, there is an exception for special kinds of types known as
//!   [Dynamically Sized Types](https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts)
//!   (DSTs). Rust's slices (e.g. `[f64]`) and string slices, `str` are
//!   prominent examples of DSTs. These are effectively implemented as
//!   "fat pointers" (i.e. they contain the pointer-address and metadata about
//!   the size of the referenced data).
//! - My immediate thought was: "why can't we do that?" It turns out that DSTs
//!   aren't a fully fledged concept. They are *extremely* restricted outside
//!   the standard library. We can *only* define a new DST by having it wrap an
//!   existing DST (namely the slice type, `[f64]`).
//! - We can't do that unless we make 1 of the following 2 restrictions:
//!     1. We force the contents of a `StatePackView` to always be contiguous
//!        (this is bad for GPU memory coalescing)
//!     2. We forbid holding a mutable `StatePackView` and an immutable
//!        `StatePackView` for a single `StatePackList` at a given time.
//!
//!   We should definitely re-evaluate this in the future, but I'm not willing
//!   to make these restrictions at this time.
//! - Any other approach for trying to implement `StatePackView` as a DST,
//!   would either:
//!     - be stupidly slow, OR
//!     - trigger undefined behavior when we create mutable & immutable
//!       `StatePackView` instances at the same time from the same
//!       `StatePackList`: you wouldn having mutable & immutable slices that
//!       include overlapping memory regions (this isn't allowed in safe or
//!       unsafe Rust)
use crate::misc::View2DUnsignedSpec;
use core::{
    num::NonZeroUsize,
    ops::{Index, IndexMut},
};

/// A read-only view of a single accumulation state
///
/// This represents a view onto a full array external array or a subregion of
/// a larger array. The lifetime parameter that parameterizes the type,
/// describes the scope of the borrow.
///
/// This is commonly constructed from either [`StatePackViewMut`],
/// [`StatePackView`], or [`AccumStateViewMut`].
///
/// See also [`AccumStateViewMut`].
pub struct AccumStateView<'a> {
    // see the comment within the declaration of AccumStateViewMut for some
    // important implementation notes pertaining to implementation (and are
    // very relevant for any kind of refactoring)
    len: NonZeroUsize,
    stride: usize,
    data: &'a [f64],
}

impl<'a> AccumStateView<'a> {
    /// Private constructor used by other types in this module
    fn internal_new(len: NonZeroUsize, stride: usize, data: &'a [f64]) -> Self {
        debug_assert!(((len.get() - 1) * stride) < data.len());
        Self { len, stride, data }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len.get()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }
}

impl<'a> Index<usize> for AccumStateView<'a> {
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index * self.stride)
    }
}

/// A read-write view of a single accumulation state
///
/// This represents a view onto a full array external array or a subregion of
/// a larger array. The lifetime parameter that parameterizes the type,
/// describes the scope of the borrow.
///
/// This is commonly constructed from [`StatePackViewMut`].
///
/// See also [`AccumStateView`].
pub struct AccumStateViewMut<'a> {
    // Someday, we may choose to implement this in terms of a pointer if we
    // wanted to be able to make multiple disjoint views onto a
    // StatePackViewMut, where at least 1 view is mutable (the docstring of
    // StatePackViewMut provides some additional detail).
    //
    // If we go that route, we probably need to employ
    // `core::marker::PhantomData` to properly track lifetimes
    // (see https://doc.rust-lang.org/nomicon/phantom-data.html)
    len: NonZeroUsize,
    stride: usize,
    data: &'a mut [f64],
}

impl<'a> AccumStateViewMut<'a> {
    /// Private constructor used by other types in this module
    fn internal_new(len: NonZeroUsize, stride: usize, data: &'a mut [f64]) -> Self {
        debug_assert!(((len.get() - 1) * stride) < data.len());
        Self { len, stride, data }
    }

    // consider returning an option rather than panicking
    pub fn from_contiguous_slice(data: &'a mut [f64]) -> Self {
        let Some(len) = NonZeroUsize::new(data.len()) else {
            panic!("can't construct an empty AccumStateViewMut");
        };
        let stride = 1;
        Self { len, stride, data }
    }

    pub fn as_view<'b>(&'b self) -> AccumStateView<'b> {
        AccumStateView {
            len: self.len,
            stride: self.stride,
            data: self.data,
        }
    }

    // todo: consider whether this should actually be a public method shipped
    // in a release
    pub fn fill(&mut self, val: f64) {
        self.data.fill(val);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len.get()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }
}

impl<'a> Index<usize> for AccumStateViewMut<'a> {
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index * self.stride)
    }
}

impl<'a> IndexMut<usize> for AccumStateViewMut<'a> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data.index_mut(index * self.stride)
    }
}

/// Represents a read-only collection of accumulation states
///
/// You should think of this as a read-only variant of [`StatePackViewMut`].
/// The internal representation of both types are as similar as possible.
///
/// # Note
/// Ideally, we would be able to totally dispose of this type... But for now
/// it serves a purpose...
pub struct StatePackView<'a> {
    // see StatePackViewMut<'a> for refactoring notes
    data: &'a [f64],
    idx_spec: View2DUnsignedSpec,
}

impl<'a> StatePackView<'a> {
    pub fn from_slice(n_state: usize, state_size: usize, data: &'a [f64]) -> Self {
        assert!(n_state > 0);
        assert!(state_size > 0);
        let idx_spec = View2DUnsignedSpec::from_shape_contiguous([state_size, n_state]).unwrap();
        if idx_spec.required_length() > data.len() {
            panic!("data doesn't hold the appropriate number of elements");
        }
        Self { data, idx_spec }
    }

    pub fn as_slice(&self) -> &[f64] {
        self.data
    }

    #[inline]
    pub fn get_state(&self, i: usize) -> AccumStateView<'_> {
        let start = self.idx_spec.map_idx2d_to_1d(0, i);
        let len = unsafe { NonZeroUsize::new_unchecked(self.state_size()) };
        AccumStateView::internal_new(len, self.idx_spec.strides()[0], &self.data[start..])
    }

    #[inline]
    pub fn state_size(&self) -> usize {
        self.idx_spec.shape()[0]
    }

    #[inline]
    pub fn n_states(&self) -> usize {
        self.idx_spec.shape()[1]
    }
}

/// Represents a read-write collection of accumulation States
///
/// This type is often used to store a binned statepack.
///
/// # Data Representation
/// This type stores accumulator states in an interleaved manner. In other
/// words, the data associated with an [`StatePackView`] instance
/// returned by [`Self::get_state`] cannot be contiguous unless the
/// [`Self::n_states`] method returns `1`.[^single_elem]
///
/// ## Benefits
/// This choice has some theoretical benefits:
/// - when updating a binned statepack with values from another binned
///   statepack, this representation would facilitate SIMD optimizations
///   on a CPU. (It could also facilitate memory-colasence if the operation
///   were parallelized on GPUs)
/// - If you implemented [`crate::Team::calccontribs_combine_apply`] for a
///   team where each member corresponds to a vector lane, you would
///   theoretically want to hold the temporary accum_states in an interleaved
///   manner (as is done by this type)
///
/// ## Complex disjoint mutable views
/// Unfortunately, this choice introduces challenges if we want to be able to
/// create 2 or more [`AccumStateViewMut`] instances that represents views on
/// separate accumulation states held by a single [`StatePackViewMut`]
/// instance. The same challenges are relevant for creating a
/// [`AccumStateViewMut`] **and** 1 or more [`AccumStateView`] instances
/// (note: there isn't any problems creating multiple [`AccumStateView`]s).
///
/// The **only** way to support the creation of disjoint mutable views is by
/// implementing [`AccumStateView`] and [`AccumStateViewMut`] in terms of raw
/// pointers, and to have the function that creates disjoint regions ensure
/// that there is no overlap.
///
/// At the time of writing, [`AccumStateView`] and [`AccumStateViewMut`], are
/// implemented in terms of slices. That implementation choice coupled with
/// the choice of to interleave the memory of accumulation state makes it
/// **IMPOSSIBLE** to construct 2 or more accumulation views at the same time,
/// when at least one of the views is mutable.
/// - The fundamental problem is that any pair of views would be represented
///   by a pair of slices that must reference overlapping memory. It is
///   impossible to construct a pair of slices that does this (when at least
///   one slice is mutable)
/// - this sentiment is reinforced by a safety note in the docstring of the
///   [`core::slice::from_raw_parts_mut`](https://doc.rust-lang.org/core/slice/fn.from_raw_parts_mut.html)
///
/// [^single_elem]: Technically, a [`StatePackView`] instance returned by
///     [`Self::get_state`] could be considered contiguous if
///     [`Self::state_size`] returns `1`. But that's only because a 1-element
///     array is always contiguous.
pub struct StatePackViewMut<'a> {
    // If we ever reimplement this in terms of a pointer (I don't think that
    // there's a benefit), we should consider whether we need to make use of
    // core::marker::PhantomData. I don't *think* it would necessary for this
    // scenario, but we should review
    // https://doc.rust-lang.org/nomicon/phantom-data.html
    data: &'a mut [f64],
    idx_spec: View2DUnsignedSpec,
}

impl<'a> StatePackViewMut<'a> {
    pub fn from_slice(n_state: usize, state_size: usize, data: &'a mut [f64]) -> Self {
        assert!(n_state > 0);
        assert!(state_size > 0);
        let idx_spec = View2DUnsignedSpec::from_shape_contiguous([state_size, n_state]).unwrap();
        if idx_spec.required_length() > data.len() {
            panic!("data doesn't hold the appropriate number of elements");
        }
        Self { data, idx_spec }
    }

    pub fn as_slice_mut(&mut self) -> &mut [f64] {
        self.data
    }

    pub fn as_slice(&self) -> &[f64] {
        self.data
    }

    pub fn as_view<'b>(&'b self) -> StatePackView<'b> {
        StatePackView {
            data: self.data,
            idx_spec: self.idx_spec.clone(),
        }
    }

    #[inline]
    pub fn get_state(&self, i: usize) -> AccumStateView<'_> {
        let start = self.idx_spec.map_idx2d_to_1d(0, i);
        let len = unsafe { NonZeroUsize::new_unchecked(self.state_size()) };
        AccumStateView::internal_new(len, self.idx_spec.strides()[0], &self.data[start..])
    }

    #[inline]
    pub fn get_state_mut(&mut self, i: usize) -> AccumStateViewMut<'_> {
        let start = self.idx_spec.map_idx2d_to_1d(0, i);
        let len = unsafe { NonZeroUsize::new_unchecked(self.state_size()) };
        AccumStateViewMut::internal_new(len, self.idx_spec.strides()[0], &mut self.data[start..])
    }

    #[inline]
    pub fn state_size(&self) -> usize {
        self.idx_spec.shape()[0]
    }

    #[inline]
    pub fn n_states(&self) -> usize {
        self.idx_spec.shape()[1]
    }

    pub fn total_size(&self) -> usize {
        self.state_size() * self.n_states()
    }
}
