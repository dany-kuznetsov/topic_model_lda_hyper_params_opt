# LDA topic model hyper params optimization

Summary

Two implementations of LDA model:
- Gensim LDA model
- Guided LDA

For each implementation there are several functions to optimize hyper parameters of LDA model in two stages:
- Stage 1 - optimize almost all params with fixed interval for number of topics
- Stage 2 - optimize anly number of topics with fixed almost all params from Stage 1 optimization

Functions are stored in `topic_model_hyper_param_opt.py` module.
Examples of their application are in notebooks.

## Notebooks and Scripts description

### Script with functions - `topic_model_hyper_param_opt.py`
This script contains a set of functions. These functions helps to 
- Preprocess texts
- Optimize hyper params of LDA topic model
- Plot the results of hyper params optimization
- Run experiments with hyper params opt


### Notebook - `gensim_lda_hyperopt_example.ipynb`
This notebook provides an example of implementation functions of two stage optimization process.
This notebook contains a pipeline with steps from reading the data to plot the results.
The proposed pipeline have several steps:
- **Read the data**.
- **Preprocessing**.
- **Preparing objects for model** - `corpus` and `id2word`.
- **Stage 1 hyper params opt** - optimize params of LDA model with fixed limited interval of number of topics. Optimal values for almost all params except number of topics are obtained as a result of this step.
- **Stage 2 hyper params opt** -  optimize number of topics with fixed other params of LDA model. Fixed other params - obtained optimal params from Stage 1. This step helps to optimize only number of topics.
- **Build a model** with optimal params
- **Plots** - params values vs. loss
- **Model description** with prints, word clouds, pyLDAvis and the most representative documents for each topic


### Notebook - `gensim_lda_experiment_example.ipynb`
A notebook provides an example of implementation experiments. Each experiment is a whole pipeline in one function which allows to:
- Preprocess data
- Create objects for model
- Optimize LDA model hyper params on Stage 1 and Stage 2
- Save trials of hyperopt in dataframe format
- Display and save plots

So there is the example how to run several experiments to choose preprocessing params of pipeline and LDA model optimal params.


### Notebook - `guided_lda_hyperopt_example.ipynb`

Guided LDA allows us to seed initial topics. Guided LDA tries to fit to these seeds as target topics with some seed_confidence which represent an allowed deviation from seeded topics.
