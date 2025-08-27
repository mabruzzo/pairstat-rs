/*!
This crate implements most of the underlying details of the
`pairstat` crate.

# Developer Guide

<div class="warning"> We may relocate this </div>

## Parallelism Overview

The primary routines in this crate have all been implemented as _binned
reductions_ in the context of a simple parallel programming model. We provide
a high-level overview of what this actually means in this section.

In the first 2 subsections, we define some general-purpose concepts &
machinery related to reductions, while ignoring parallelism. Then, we briefly
connect the discussion to two-point statistics, before we introduce our
parallel programming model. and finally describing the two primary flavors
of the parallel algorithms that arise for implementing binned reductions.

### Simple Reductions

Let's start with the concept of a simple reduction (and totally ignore
parallelism for the moment).

For the unitiated, a reduction is a common abstraction that describes the
algorithm of repeatedly applying an operation to convert multiple values down
to a single result. This comes up in parallel programming because the repeated
operation is associative and commutative, in a mathematical
sense.[^finite_precision]

#### Fleshing things out

<font color="red"> We should consider whether there is a better way to present this information! </font>

**Key Takeaway:** We implement a reduction in terms of a [`Reducer`] type
that stores incremental contribution, in a separately tracked `accum_state`,
from each [`Datum`] instance that is produced from
a __data source__.

For the sake of concreteness, and to introduce some basic terminology, let's
describe a hypothetical calculation that we consider a reductions. This
calculation we compute some statistic to describe some property of multiple
pieces of data:
* We represent each "piece of data" as an instance of the
  [`Datum`] type.
* We generally say that we produce  [`Datum`]
  instances from a __data source__.  Importantly, the term, _data source_, is
  an abstract concept; corresponding types and functions depend on context.
* We're only interested in calculations that can be implemented
  as a [one-pass algorithm](https://en.wikipedia.org/wiki/One-pass_algorithm)
  (in principle, we could add support for multi-pass algorithms).
  * as the algorithm considers each [`Datum`]
    instance, it repeatedly applies some operation that modifies some temporary
    state that represents the accumulated contributions from all instances
    considered "so far." (In the context of two-point statistics, a `Datum`
    corresponds to a value computed from a pair of points.)
  * importantly, we require that this logic is mathematically associative
    and commutative (i.e. we can work on arbitrary subsets of the data in an
    arbitrary order and merge the results together later)
  * examples of simple statistics that can be computed include (but aren't
    limited to) sums, means, variance, minima, maxima, etc.
* throughout the documentation, we refer to this intermediate state for
  accumulating contributions as an `accum_state`.
* we encode the logic for using a [`Datum`] to update
  an `accum_state` and for merging separate `accum_state`s in a type that we
  call a _Reducer_. A _Reducer_ type
  * always implements the [`Reducer`] trait.
  * is __ALWAYS__ tracked separately from the `accum_state`\
  * also encodes logic for initializing/resetting an `accum_state` and for
    computing the final result(s) from an `accum_state`


### Binned Reduction

*(Let's continue ignoring parallelism, for the moment)*

This crate is focused on computing "Binned Reductions." The basic
premise is that each [`Datum`] instance taken from a
data source is partitioned into a separate bin. In other words, we generate a
set of `(bin index, datum)` pairs from the data source. In some parts of the
code, we represent this pair as a [`BinnedDatum`]
instance. During the _Binned Reduction_, we track separate `accum_state`s for
each bin and a datum only contributions to the `accum_state` in the
corresponding bin.

We generally describe a collection of `accum_state`s as "statepack." We use
the term `binned_statepack` when each `accum_state` in a `statepack`
corresponds to a distinct bin index. Within the codebase, we often represent
a statepack as a [`StatePackViewMut`].

### Intermezzo: Relating back to 2-points statistics

Let's briefly pause and relate these concepts back to two-point statistics.
Let's be concrete and imagine that we want to compute the isotropic density
correlation function using information from 2 input datasets. Each dataset
holds density measurements associated with spatial position. Importantly, the
correlation function specifies correlation as function of distance for some
specified set of "distance" bins.
- we would characterize the combination of both datasets as the _data source_.
- from the _data source_, we would generate a (bin index, datum) pair for each
  unique measurement pair (we draw 1 measurement from each dataset). For a given
  measurement pair:
  - the bin index is the index of the specified "distance" bin that
    corresponding to the distance between the spatial positions of each
    measurement in the pair.
  - The datum, is composed of a value and a weight. In this case:
    - the datum's value is the product of both density measurements
    - the datum's weight is the product of both measurments' weights. In the
      common case where a measurement doesn't have a weight, you can assume
      that all measurements have a weight of 1.
  - In our binned reduction, we would use a _Reducer_ type designed for
    computing a weighted mean.

It's pretty straight-forward to understand that the above discussion easily
generalizes to the case where you want to compute auto-correlation for all
measurements from a single dataset (we just have to take a little care with
the way we identify all unique measurement pairs).

More generally, our implementation actually considers vector-quantities like
velocity, rather than scalar quantities like density. But, it isn't that much
more complicated

### Parallelism: Teams

Now, let's shift gears and start to consider parallelism. At a high-level, our
programming model, considers 2 levels of parallel processing: (i) a
fine-grained highly synchronous/collaborative level and (ii) a coarse-grained
level, with minimal collaboration (i.e. there is some basic coordination at the
very start and at the very end of a task).

This section primarily focuses on the fine-grained parallelism. In fact, let's
consider this fine-grained parallelism in a vacuum, and forget about the
coarse-grained parallelism for now; we'll touch on coarse-grained parallelism
at the end of this section.

To model fine-grained parallelism, we introduce a __team__ abstraction. A team
is composed of 1 or more members, and the members of a team work together in a
tightly-coupled, synchronous manner to collaboratively complete a single task,
or "unit of work", at a time.
- your mental model should be that members of a team perform a operation at
  a time in lockstep with each other. For certain operations, team members may
  need to be idle
- how does this map to hardware?
  - The simplest team is composed of 1 member. This can be trivially
    implemented on a single CPU
  - It should be obvious that the idea of a team maps nicely to SIMD. In this
    case each team member maps to a SIMD lane on a CPU.
  - on a GPUs, members of a team map nicely to members of a warp
    - while it's simplest to have the number of team members match the number
      of threads in a warp, we do have flexibility (e.g. map to half a warp,
      map to 2 warps).
    - technically GPU threads in a Warp have a lot more flexibility, but we
      target this sort of parallelism for maximum portability.
  - we also have the freedom to implement a "team" in other ways to test
    parallelism

The basic premise is to implement different parallelism backends. In a given
backend, we represent a "team" with a type that implements the
[`Team`] trait.

Finally, let's briefly reconsider the coarser-level parallelism. The idea is
to use multiple teams, where each team operates independently. This is
straightforward to support:
- You need to properly partition "units of work" so that separate teams work
  on separate parts of the reduction. Each team would effectively track its
  own dedicated `binned_statepack`.
- At the very end, you need to consolidate the contributions from every
  `binned_statepack`.

<div class="warning">

At the moment, this crate doesn't fully implement the deeper level of
parallelism. Furthermore, the "team" abstraction is somewhat "leaky" (you need
to take some "special" steps to leverage SIMD). Nevertheless, it's useful to
develop our algorithms using this abstraction, because it forces us to make
algorithmic choices that are compatible with the deeper parallelism, should we
choose to use it in the future (i.e. there are many ways to implement these
algorithms that aren't compatible).

</div>

### Parallelizing Binnned Reductions

Let's briefly recap: this crate implements computes binned reductions. As
we've already noted, a binned reduction focuses on updating binned
`accum_state`s. It's accomplished by generating (bin-index, datum) pairs
from a data-source and repeatedly performing a mathematically associative and
commutative operation to update the appropriate `accum_state` from each datum.

How do we parallelize this? In practice, we decompose the calculation into 1
or more "units of work." Each "unit of work" consists of generating a unique
subsets of the (datum, bin-index) pairs and using them to update the
appropriate `accum_state`s. We lightly touched on this before, but it's worth
repeating. All members of a given team work together on a single "unit of work"
at a time. So, if a team is assigned 2 "units of work," the team works together
until the first unit is completely done before they all move onto the next
unit.

The way that we define a "unit of work" depends on the nature of the data
source. We will talk about that in the next section.

### Kinds of Parallel Binned Reductions

In this crate, Parallel Binned Reductions fall into 2 categories related to
the nature of the data source and the way that we define "units of work." (You
can think of these as extreme cases on a continuum of algorithms). We call
these categories _"Nested"_ and _"Batched"_.

1. _"Nested"_ reductions come up when we know ahead of time that each datum
   generated as part of a "unit of work" will share a common bin index.
   - When a team starts working on a "unit of work", each member gets its own
     temporary `accum_state`
   - Each team member then generates a unique subset of datum instances
     associated with the "unit of work" and stores the contributions to the
     overall reduction within its temporary `accum_state`
   - Once all the members are done doing this, they merge the contributions
     from their respective temporary `accum_state`s and one of the members
     then use this total contribution to update the appropriate entry in
     the team's `binned_statepack`.

2. _"Batched"_ reductions are much more general (albeit less efficient). In
   this case, we know nothing about the relationships each datum and the
   associated bin index that will be generated in the unit of work. Thus,
   the best we can do is adopt a "batching strategy:"
   - each team member independently generates a unique subset of the
     (bin index, datum) pairs associated with the current "unit of work".
   - these pairs are all gathered in a collection-pad
   - then one of the members serially iterates through the full batch of
     (bin index, datum) pairs and sequentially updates the team's
     `binned_statepack`.

[^finite_precision]: In practice, the finite precision used to represent
    numbers means that these operations are not, strictly speaking,
    associative. This is especially relevant for floating point numbers.

*/

#![cfg_attr(not(test), no_std)] // (we allow std for testing)
#![deny(rustdoc::broken_intra_doc_links)]

mod bins;
mod misc;
mod parallel;
mod reduce_utils;
mod reducer;
mod state;
mod twopoint;

// I'm not really sure we want to publicly expose reduce_sample, but for now,
// we expose it to support testing...
pub mod reduce_sample;

pub use bins::*;
pub use misc::View3DSpec;
pub use parallel::{
    BinnedDatum, Executor, ReductionSpec, StandardTeamParam, Team,
    fill_single_team_binned_statepack,
};
pub use reduce_utils::{
    merge_full_statepacks, reset_full_statepack, serial_consolidate_scratch_statepacks,
};
// TODO: before 1.0, consider whether we really want to expose all of the
//       following constructs... (especially Datum)
pub use reducer::{
    Comp0Histogram, Comp0Mean, ComponentSumHistogram, ComponentSumMean, Datum, OutputDescr,
    Reducer, ScalarHistogram, ScalarMean, ScalarizeOp,
};
pub use state::{AccumStateView, AccumStateViewMut, StatePackView, StatePackViewMut};
pub use twopoint::{
    common::PairOperation,
    spatial::{CartesianBlock, CellWidth},
    twopoint_cartesian::TwoPointCartesian,
    unstructured::{TwoPointUnstructured, UnstructuredPoints},
};
