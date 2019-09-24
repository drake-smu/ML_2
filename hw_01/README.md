
# Assignment 1: Classification Tuning

>Carson Drake  
DS7335 | Fall 2019

## Description

```
# 1. write a function to take a list or dictionary of clfs and hypers ie use
#    logistic regression, each with 3 different sets of hyper parameters for
#    each
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf
#    and parampters settings
# 5. Please set up your code to be run and save the results to the directory
#    that its executed from
# 6. Collaborate to get things
# 7. Investigate grid search function
```

## Execution/Planning

To standardize model methods and tuning across various models, I implemented a
several wrapper modules for `classifiers` and `grids`.  

`models.classifiers` provides generalized wrapper classes for a few of sklearn's
classification model classes. By establishing a base `classifier` class we are
able to standardise the namespace of attributes and methods across models which
helps when iterating through arrays/collections of grid searches.  

`models.grids` provides a `Grid` class that is initiated with a single base
model and list or dictionary of hyperparam options. Based on these param
options, the `Grid` object generates all specified permutations of model
parameters. The object can then create collection of tuned models for each set
of hyperparams. These models are then trained and evaluared on the same data.
The `GridSearch` class in `models.grids` serves as a collector of `Grid`
objects. Again, the normalized namespace allows us to easily iterate through
grids of different models using different hyperparams without needing to keep
a mental map of all skleanr configs.

## Results/Demo Output

`assignment01.py` utlizes our model modules to satisfy [Steps 1-7 (described above)](#Description). Given that some of the steps are more openended, some steps are combined while others were
repurposed to demonstrate different aspects of functionality.

```bash
python assignment01.py
```
Will take you through the defined "steps" of the demonstration. As the steps are executed,
progress is written out to the `std.out`. More verbose results/summarys are written to output
files located in `outputs/`. Visualizations and serialized model objects can also be located
in `outputs/`.  

**Important: when running `assignment01.py`, if there is already an existing target model
saved in `outputs/` with the same name (default is `pt5/pickle`), then `step_four` will
load the stored model instead of training evaluating grids searches from scratch.**

## Going Forward

As with most repos...there is still much work to be done.

**TODO'S**

- [ ] 🔥🔥 Multiprocess functionality to boost training performance
- [ ] Expand on the scope of available models
  - [ ] More classifier options
  - [ ] support for continuous models
- [ ] 🔥🔥 Replace Sklearn references with non bloated self made implementations
- [ ] 🔥🔥 Improve/extend model evaluation methods. (ie other measures than just
    accuracy, better outputs, more output configs.)
- [ ] Add some sort of ledgend to visualizations.
- [ ] Try to add heat maps (other visualizations).
- [ ] Possibly isolate/group results per param adjusted
- [ ] 🔥🔥 Add support for ranges of values (with step config) in addition to
explicate values.
