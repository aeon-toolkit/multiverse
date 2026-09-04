# Runtime

**Coming soon.** This page will hold runtime comparisons across the Multiverse
estimators. 

`tsml-eval` writes, per classifier, dataset and resample:

- `fit_time` and `predict_time`, in seconds;
- `benchmark_time`, the time that machine took to sort 1,000 seeded random arrays of
  20,000 elements;

They are in the raw prediction files. 

**The runs are spread across different hardware.** Multiverse results have been produced
on Southampton's IRIDIS GPU partitions with H200 and A100 cards. GPU jobs in our configurations are allocated two CPUs each. A fit time from one
partition and a fit time from another are two different measurements that happen to share
a unit. CPU experiments were run on UEA's HALI cluster.

**GPU and CPU methods are not on one axis.** For the deep learners nearly all the work is
on the accelerator and the host CPU mostly feeds batches; for the classical ensembles
there is no accelerator at all and the time scales with the cores allocated. Comparing
them measures the hardware at least as much as the algorithm, and the ratio moves when
either side changes. 

**Wall-clock contains things that are not the algorithm.** Queueing, data loading, and
retries: our controllers escalate a job's memory request from 64 GB to 128 GB after a
failure, so an elapsed time can include a dead run at the lower ceiling.

**Training time is a hyperparameter, not a property of a method.** A deep learner's fit
time is close to linear in the number of epochs, and the epoch count is a choice. Two
faithful ports of the same paper can differ several-fold on time because the authors
picked 500 epochs and the toolkit's default is 2000. Early stopping and best-epoch
selection move it again. None of that is a fact about the architecture.

We can give a crude comparison with the results we have for the CPU and GPU groups independently. However, 
an experiment to characterise the run time complexity and a function to estimate expected run time for a specific
configuration in controlled environments would be more useful. 
