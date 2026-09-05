# Memory

**Coming soon.** This page will hold memory comparisons across the Multiverse
estimators.

**We have not yet structured an experiment to compare memory.** As with
[runtime](runtime.md), every run behind the results in this repository was set up to
measure predictive performance. A job was given whatever memory ceiling got it to
finish, on whatever node was free, with whatever core count came with the partition,
and those were never held constant across estimators because nothing depended on it.
The peak figures that came out are a by-product of that, not a measurement anyone
designed.

So nothing is published here yet, deliberately. The rest of this page records what a
memory comparison would have to fix, and why the figures we already hold cannot stand
in for one.

## What is recorded

`tsml-eval` writes one `memory_usage` value per classifier, dataset and resample: the
peak memory observed during `fit`. It is in the raw prediction files, but it is only a
proxy for the memory footprint of a classifier.

## Why the figures we hold cannot stand in

Each of these is a condition a memory experiment would have to fix, and that these runs
left free.

**Peak process memory is not the model's memory.** It includes the interpreter, every
imported library, the loaded dataset and any transient copies made along the way.
Importing TensorFlow or PyTorch alone accounts for a large fixed cost before a single
weight is allocated, so a small model in a heavy framework can report more than a large
model in a light one. Without subtracting a per-framework baseline the figure mostly
ranks frameworks.

**The dataset can dominate the model.** Multiverse contains problems like EigenWorms at
17,984 timepoints and FaceDetection at 5,890 training cases. For an estimator with a
small parameter count, most of the peak is the data and its copies, which says
something about the problem rather than the method.

**Host and device memory are different quantities, and only one is recorded.** A model
doing its work on a GPU can look inexpensive by peak host memory while occupying tens of
gigabytes of device memory that nothing here measures. The two are not interchangeable
and cannot be added.

**Framework allocators do not report what the model needs.** PyTorch's caching allocator
and TensorFlow's default of reserving most of the visible GPU both hold memory they are
not using, so a naive reading measures the allocator's policy rather than the model's
requirement. Getting a meaningful device figure means enabling TensorFlow's memory
growth and reading PyTorch's allocated rather than reserved totals, neither of which
these runs did.

**Failures censor the measurement.** Where an estimator ran out of memory we do not have
a peak, we have a lower bound and a ceiling. Both kinds appear in
`results/multiverse/missing_results.csv`: ConvTran hit CUDA out of memory on Alzheimers,
EigenWorms and PhotoStimulation, a device-side limit; FreshPRINCE hit OOM at 128 GB on
FaceDetection, FordChallenge and Skoda after eight attempts, a host-side one. Those are
the cases where memory mattered most, and they are exactly the cases with no number. A
table built only from successful runs is a survivorship-biased view of memory use.

**Memory scales with the resources granted.** The classical ensembles allocate per
thread, so their peak moves with the cores allocated, which varied by partition. Our
controllers also escalate a job's request from 64 GB to 128 GB after a failure, so
different runs of the same estimator saw different ceilings.

**Peak is timing-dependent.** Python's peak resident memory depends on when garbage
collection happens to run and on whether an allocator returned pages to the operating
system. Repeats of an identical run differ for reasons that have nothing to do with the
method.

## What a fair comparison would need

- All compared estimators on the same node, with cores and the memory ceiling fixed and
  recorded.
- A per-framework baseline measured and subtracted, so the figure is the model's cost
  rather than the cost of importing its library.
- Host and device peaks reported separately, never summed, with TensorFlow memory growth
  enabled and PyTorch read via its allocated totals.
- `predict` measured as well as `fit`. Deployment cost is a separate question from
  training cost, and the ranking is not the same on both.
- Failures reported alongside successes, as censored observations with the ceiling that
  was in force, rather than dropped.
- Repeated runs, since peak varies between identical repeats.
- Asymptotic space complexity in the number of cases, series length and channels stated
  beside the measured peaks, so a reader can tell whether a figure will hold at a
  different scale.

Until most of that is in place, this page stays empty. For the same reason the
[leaderboard](leaderboard.md) carries no memory column.
