# Runtime

**Coming soon.** This page will hold runtime and memory comparisons across the
Multiverse estimators.

**We have not yet structured an experiment to compare runtime.** Every run behind the
results in this repository was set up to measure predictive performance. Which partition
a classifier was queued on, how many cores it was given, how many epochs it trained for,
whether a job was retried at a higher memory ceiling: all of those were chosen to get
accurate results out at a reasonable cost, and none were held constant across estimators
because nothing depended on it. The timings that came out are a by-product of that, not
a measurement anyone designed.

So nothing is published here yet, deliberately. Comparing runtime needs its own
experiment, with the conditions below fixed in advance, and we have not run one. The
rest of this page records what those conditions are, and why the figures we already hold
cannot stand in for them.

## The measurements exist

Every run already records timings. `tsml-eval` writes, per classifier, dataset and
resample:

- `fit_time` and `predict_time`, in seconds;
- `memory_usage`, the peak memory during `fit`;
- `benchmark_time`, the time that machine took to sort 1,000 seeded random arrays of
  20,000 elements.

They are in the raw prediction files. What this repository ingests under `results/` is
only the accuracy-style measures, one file per metric, so the timings have not been
brought across yet. That is a small piece of work; the reason it has not been done is
below, not the effort.

## Why the figures we hold cannot stand in

Each of these is a condition a timing experiment would have to fix, and that these runs
left free.

**The runs are spread across different hardware.** Multiverse results have been produced
on GPU partitions with H200 and A100 cards and on CPU-only nodes, with different core
counts. GPU jobs in our configurations are allocated two CPUs each. A fit time from one
partition and a fit time from another are two different measurements that happen to share
a unit.

**GPU and CPU methods are not on one axis.** For the deep learners nearly all the work is
on the accelerator and the host CPU mostly feeds batches; for the classical ensembles
there is no accelerator at all and the time scales with the cores allocated. Comparing
them measures the hardware at least as much as the algorithm, and the ratio moves when
either side changes. A statement like "X is 40 times faster than Y" is, in this setting,
a statement about a purchasing decision.

**Wall-clock contains things that are not the algorithm.** Queueing, data loading, and
retries: our controllers escalate a job's memory request from 64 GB to 128 GB after a
failure, so an elapsed time can include a dead run at the lower ceiling.

**Training time is a hyperparameter, not a property of a method.** A deep learner's fit
time is close to linear in the number of epochs, and the epoch count is a choice. Two
faithful ports of the same paper can differ several-fold on time because the authors
picked 500 epochs and the toolkit's default is 2000. Early stopping and best-epoch
selection move it again. None of that is a fact about the architecture.

**Memory is measured in one place and spent in another.** `memory_usage` is peak host
memory during `fit`. A model doing all its work on a GPU can look inexpensive by that
measure while occupying tens of gigabytes of device memory, which the figure never sees.
Peak host memory and peak device memory are different quantities and only one is
recorded.

**Which device a run actually used is not reliably recorded.** In the version of the
experiment tooling used for these runs, the device description inspects TensorFlow only,
so a PyTorch estimator reports CPU whether or not it ran on a GPU. Any timing table built
from those records has to have its device column reconstructed from the job
configuration rather than trusted as written.

## What a fair comparison would need

- The compared estimators run on the same hardware, or CPU timings normalised by
  `benchmark_time`, which exists for exactly this purpose. There is no equivalent
  normaliser for GPU work.
- Thread and core counts fixed and recorded, since the classical methods scale with them.
- `fit` and `predict` reported separately. They answer different questions: fit time is
  the cost of research, predict time is the cost of deployment, and the ranking is not
  the same on both.
- Repeated runs. Timings vary far more between repeats than accuracy does, especially on
  shared nodes.
- Asymptotic complexity in the number of cases, series length and channels reported
  beside the measured times, so a reader can tell whether a result will hold at a
  different scale.
- The device stated per run, from the job configuration.

Until most of that is in place, this page stays empty. For the same reason the
[leaderboard](leaderboard.md) carries no fit time, predict time or memory columns.
