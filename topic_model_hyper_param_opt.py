import numpy as np
import pandas as pd
import math
import json

import matplotlib.pyplot as plt

import pickle
import re
import guidedlda

import scipy

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from sklearn.feature_extraction import text


def preprocess_doc(doc, lang="german", sw=True, custom_stop_words=[]):
    '''
    Preprocessing a document with
        lower text 
        replace umlauts
        filter stop words
        and custom stop words
        lemmatization
    '''
    doc_low = doc.lower()

    # keep only text
    if lang == "english":
        doc_az = re.sub("[^a-z]", " ", doc_low)  # keep a-z
    elif lang == "german":
        doc_az = re.sub("[^a-zäöüÄÖÜß]", " ", doc_low)

        umlauts = {
            "Ä": "Ae",
            "ä": "ae",
            "Ö": "Oe",
            "ö": "oe",
            "Ü": "Ue",
            "ü": "ue",
            "ß": "ss",
        }

        for uml in umlauts:
            doc_az = doc_az.replace(uml, umlauts[uml])
    # tokenization
    doc_token = gensim.utils.simple_preprocess(str(doc_az))

    # filter stopwords
    if sw:
        if lang == "english":
            stop = stopwords.words("english")
            stop.extend(['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people', 
                         'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said'])
            stop = text.ENGLISH_STOP_WORDS.union(stop) # added row of code to extend stop words
        elif lang == "german":
            stop = stopwords.words("german")
            stop.extend([
                "der", "und", "nicht", "ich", 
#                 'fuer', 'ueber', 'mehr', 'ab', 'schon', # added as stop words from 10kGNAD iter 1
#                 'wurde', 'worden', 'wurden' # added as stop words from 10kGNAD iter 2
            ]) 
        # add custom stop words    
        stop.extend(custom_stop_words)
        doc_token = [token for token in doc_token if token not in stop]
    lemmatizer = WordNetLemmatizer()
    doc_lem = [lemmatizer.lemmatize(token) for token in doc_token]

    return doc_lem




from sklearn.feature_extraction.text import CountVectorizer

def create_document_term_matrix_from_texts_series(
    texts_series,
    ngram_range=(1, 1),
    index_for_dtm=None,
    vocabulary_cut_offs={
        'min_df': 0.001,
        'max_df': 0.999
    }):
    '''
    Creates document_term_matrix from the series of previously preprocessed texts.
    If index_for_dtm == None then index will not be added to df.
    '''
    # If an element of series is solid string then just vectorize this
    if type(texts_series.iloc[0]) == str:
        texts_series = texts_series
    # If an element of series is a list of strings 
    # then create a solid string from these words
    elif type(texts_series.iloc[0]) == list:
        texts_series = texts_series.apply(lambda x: " ".join(x))
    else:
        raise ValueError('The elemets of series must be string or list type object')
    
    vectorizer = CountVectorizer(
        ngram_range=ngram_range, 
        min_df=vocabulary_cut_offs['min_df'], 
        max_df=vocabulary_cut_offs['max_df']
    )
    data_vect = vectorizer.fit_transform(texts_series)
    data_dtm = pd.DataFrame(data_vect.toarray(), columns=vectorizer.get_feature_names())
    if index_for_dtm is not None:
        data_dtm.index = index_for_dtm
    return data_vect, vectorizer, data_dtm




# Define the function of single optimization step
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimization_hyper_params_gensim_ldamodel(
    corpus,
    id2word,
    param_space,
    param_dict={},
    passes_param=2, 
    iterations_param=51,
    random_seed=420,
    max_eval_param=100,
    timeout_param=600,
    metric_to_optimize='coherence'):
    
    '''
    Optimize hyperparameters of LDA model from param_space 
    with selected fixed parameters of LDA model from params_dict.
    
    max_eval_param and timeout_param - 
        Algorithm stops when reach at least one of these conditions.
    
    Parameters
    ----------
    corpus : gensim.matutils.Sparse2Corpus
    id2word : dict
    param_space : dict
        Parameters space for optimization.
    params_dict : dict, optional
        Additional fixed parameters for LDA model.
        These params must not be in param_space
    seed : int, optional
        Random seed as a parameter of LDA model
    max_eval_param : int, optional
        Number of iterations of hyper param optimization
    timeout_param : int, optional
        Number of seconds to early stopping param optimization
    metric_to_optimize : str, optional
        Choose the metric to optimize:
        'coherence', 'perplexity'
    
    Returns
    -------
    Returns a tuple (result, trials)
    result : dict
        Optimal values of parameters from param_space
    trials : hyperopt.base.Trials
        trials.trials - List of dictionaries with info of
        optimization iterations
    
    Raises
    ------
    ValueError
        If metric_to_optimize is not a possible value
    '''
    for key in list(param_space.keys()):
        try:
            param_dict.pop(key)
        except:
            pass
    
    def optimize_lda(params, param_dict=param_dict,
                     corpus=corpus, id2word=id2word, 
                     passes_param=passes_param, iterations_param=iterations_param,
                     random_seed=random_seed, metric=metric_to_optimize):
        '''
        Function to minimize in hyper parameter optimization
        metric: 'coherence', 'perplexity'
        '''
        model_lda = gensim.models.LdaModel(
            corpus=corpus, id2word=id2word,
            passes=passes_param, iterations=iterations_param,
            random_state=random_seed,
            **param_dict, **params
        )
        # To check model's param changing over trials
        # print('iterations, decay, offset, num_topics', end=', ')
        # print(model_lda.iterations, model_lda.decay, model_lda.offset, model_lda.num_topics)
        
        if metric=='perplexity':
            metric_value = abs(model_lda.log_perplexity(corpus))
        elif metric=='coherence':
            coherence_lda = gensim.models.CoherenceModel(model=model_lda, corpus=corpus, dictionary=id2word, coherence='u_mass').get_coherence()
            metric_value = abs(coherence_lda)
        else:
            raise ValueError('Choose the metric_to_optimize of the function')
        return metric_value
    
    from functools import partial
    optimization_function = partial(optimize_lda)
    trials = Trials()
    result = hyperopt.fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=max_eval_param,
        timeout=timeout_param, # in seconds
        trials=trials,
    )
    
    return result, trials





# Define lists which represent an optimization space for Stage 1.
# They are defined this way because
# trails_of_hyper_opt_to_dataframe function should have access to these lists
# to convert indexes of params values in trials object to values from lists.

alpha_values_list = [ 
    0.5, 0.4, 0.3, 0.2, 0.1, 
    0.08, 0.06, 0.04, 0.02, 0.01, 
    0.005, 0.001, 0.0005, 0.0001
]
decay_values_list = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.55, 0.51, 0.50001]
offset_values_list = [1, 4, 16, 64, 256, 1024]
eta_values_list = [None, 'auto']
gamma_threshold_values_list = [
    0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 
    0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00005, 0.00001
]
minimum_probability_values_list = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]





def stage_1_optimization_gensim_ldamodel(
    corpus, 
    id2word,
    metric_to_optimize='coherence',
    stage_1_params={
        'num_of_topics_min': 14,
        'num_of_topics_max': 16,
        'passes_param': 10,
        'iterations_param': 50,
        'max_eval_param': 1000,
        'timeout_param': 86400, 
    },
    param_dict={},
    random_seed=420):
    
    # Define some constant values of Stage 1 optimization

    # Values lists for optimization space.
    
    # THESE LISTS ARE DEFINED ABOVE FOR ALL THIS MODULE
    # SO THAT FUNCTION WILL READ THESE LISTS DIRECTLY FROM THE MODULE

    # Alternative list for alpha: ['symmetric', 'asymmetric', 'auto']
    # alpha_values_list = [ 
    #     0.5, 0.4, 0.3, 0.2, 0.1, 
    #     0.08, 0.06, 0.04, 0.02, 0.01, 
    #     0.005, 0.001, 0.0005, 0.0001
    # ]
    # decay_values_list = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.55, 0.51, 0.50001]
    # offset_values_list = [1, 4, 16, 64, 256, 1024]
    # eta_values_list = [None, 'auto']
    # gamma_threshold_values_list = [
    #     0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 
    #     0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00005, 0.00001
    # ]
    # minimum_probability_values_list = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    # Define params space with params lists of values above
    param_space_stage1 = {
        'num_topics': scope.int(
            hp.quniform('num_topics',
                        stage_1_params['num_of_topics_min'],
                        stage_1_params['num_of_topics_max'],
                        1)
        ),
        'alpha': hp.choice('alpha', alpha_values_list),
        'decay': hp.choice('decay', decay_values_list),
        'offset': hp.choice('offset', offset_values_list),
        'eta': hp.choice('eta', eta_values_list),
        'gamma_threshold': hp.choice('gamma_threshold', gamma_threshold_values_list),
        'minimum_probability': hp.choice('minimum_probability', minimum_probability_values_list),
    }
    
    # Stage 1 optimization with almost all params of the LDA model.
    result_stage_1, trials_stage_1 = optimization_hyper_params_gensim_ldamodel(
        corpus=corpus,
        id2word=id2word,
        param_space=param_space_stage1,
        param_dict=param_dict,
        passes_param=stage_1_params['passes_param'], 
        iterations_param=stage_1_params['iterations_param'],
        random_seed=random_seed,
        max_eval_param=stage_1_params['max_eval_param'],
        timeout_param=stage_1_params['timeout_param'],
        metric_to_optimize=metric_to_optimize
    )
    
    # Params from optimization stage 1 without num_topics. Fix them for Stage 2.
    params_optimal_stage_1 = {
        'alpha': alpha_values_list[result_stage_1['alpha']],
        'decay': decay_values_list[result_stage_1['decay']],
        'offset': offset_values_list[result_stage_1['offset']],
        'eta': eta_values_list[result_stage_1['eta']],
        'gamma_threshold': gamma_threshold_values_list[result_stage_1['gamma_threshold']],
        'minimum_probability':  minimum_probability_values_list[result_stage_1['minimum_probability']],
    }
    
    return params_optimal_stage_1, trials_stage_1





def stage_2_optimization_gensim_ldamodel(
    corpus, 
    id2word,
    params_optimal_stage_1,
    metric_to_optimize='coherence',
    stage_2_params={
        'num_of_topics_min': 10,
        'num_of_topics_max': 100,
        'passes_param': 10,
        'iterations_param': 50,
        'max_eval_param': 100,
        'timeout_param': 86400, 
    },
    random_seed=420):
    
    # Define some values of Stage 2 optimization

    # Param space for Stage 2.
    param_space_stage2 = {
        'num_topics': scope.int(
            hp.quniform('num_topics', 
                        stage_2_params['num_of_topics_min'], 
                        stage_2_params['num_of_topics_max'], 
                        1)
        )
    }
    
    # Stage 2 optimization with fixed optimal params from Stage 1.
    result_stage_2, trials_stage_2 = optimization_hyper_params_gensim_ldamodel(
        corpus=corpus,
        id2word=id2word,
        param_space=param_space_stage2,
        param_dict={**params_optimal_stage_1},
        passes_param=stage_2_params['passes_param'], 
        iterations_param=stage_2_params['iterations_param'],
        random_seed=random_seed,
        max_eval_param=stage_2_params['max_eval_param'],
        timeout_param=stage_2_params['timeout_param'],
        metric_to_optimize=metric_to_optimize
    )
    
    # Define optimal params values as joined dictionary from both Stages
    optimal_params_result = params_optimal_stage_1
    optimal_params_result.update(result_stage_2)
    
    return optimal_params_result, trials_stage_2





# Define a function for two stage optimization of LDA topic modelling.
# Stage 1 - optimize params of LDA model with fixed number of topics.
# Then take these optimal params of Stage 1 and fix them for Stage 2.
# Stage 2 - optimize number of topics with fixed other params of LDA model.
def two_stage_optimization_gensim_ldamodel(
    corpus, 
    id2word,
    metric_to_optimize='coherence',
    stage_1_params={
        'num_of_topics_min': 14,
        'num_of_topics_max': 16,
        'passes_param': 10,
        'iterations_param': 50,
        'max_eval_param': 1000,
        'timeout_param': 86400, 
    },
    stage_2_params={
        'num_of_topics_min': 10,
        'num_of_topics_max': 100,
        'passes_param': 10,
        'iterations_param': 50,
        'max_eval_param': 100,
        'timeout_param': 86400, 
    },
    random_seed=420):
    
    # Stage 1 optimization with almost all params of the LDA model.
    params_optimal_stage_1, trials_stage_1 = stage_1_optimization_gensim_ldamodel(
        corpus=corpus, 
        id2word=id2word,
        metric_to_optimize='coherence',
        stage_1_params=stage_1_params,
        random_seed=420
    )
    
    # Stage 2 optimization with fixed optimal params from Stage 1.
    optimal_params_result, trials_stage_2 = stage_2_optimization_gensim_ldamodel(
        corpus=corpus, 
        id2word=id2word,
        params_optimal_stage_1=params_optimal_stage_1,
        metric_to_optimize='coherence',
        stage_2_params=stage_2_params,
        random_seed=420
    )
    
    return optimal_params_result, trials_stage_1, trials_stage_2




import sys
def trails_of_hyper_opt_to_dataframe(trials_object):
    '''
    This function takes hyperopt object with trials
    and converts this object to dataframe.

    It adds columns with params values and function loss values for each trial.
    '''
    # Convert list of dictionaries to df
    df_with_trials = pd.DataFrame(trials_object.trials)

    # But we need to make columns of params and results from dictionaries.
    # Take param names of trials and possible result outputs
    list_of_params_in_trials = list(df_with_trials['misc'][0]['vals'].keys())
    list_of_result_in_trials = list(df_with_trials['result'][0].keys())

    # Iterate these two dictionaries to create columns
    for param_now in list_of_params_in_trials:
        df_with_trials[param_now] = df_with_trials['misc'].apply(
            lambda x: x['vals'][param_now][0]
        )
    for result_now in list_of_result_in_trials:
        df_with_trials[result_now] = df_with_trials['result'].apply(
            lambda x: x[result_now]
        )
    # Iterate only columns with params to convert thier indexes to values.
    for param_now in list_of_params_in_trials:
        try:
            # num_topics param is not index,
            # so converting to values is not needed.
            if param_now == 'num_topics':
                continue
            # Param space for Stage 1 was defined as combination of values lists.
            # These lists were defined above (e.g. alpha_values_list).
            # So values_list variable is defined as one of these lists.
            values_list = getattr(sys.modules[__name__], str(str(param_now) + '_values_list'))
            # Convert indexes of params columns to their values
            # by replacing index with one of the value of list with this index.
            df_with_trials[param_now] = df_with_trials[param_now].apply(
                lambda x: values_list[x]
            )
        except Exception as e:
            print(e)
    
    return df_with_trials




import seaborn as sns
def plot_loss_vs_params(
    trials_stage_1_df,
    quantile_of_loss=1.0,
    ncols_plot=3,
    file_path='C:/DAN/t_systems/topic_modelling_project/experiment_feedback/' + 'test.png',\
    note_text=' ',
    save_plot=True,
    show_plot=True):
    
    # We will show on plot only trials which are less than some quantile loss of these trails
    median_loss = np.median(trials_stage_1_df['loss'])
    trails_with_loss_less_than_quantile = trials_stage_1_df[
        trials_stage_1_df['loss'] <= trials_stage_1_df['loss'].quantile(quantile_of_loss)
    ]

    # Choose the list of parameters to show on plot 
    optimal_params_keys_list = list(trials_stage_1_df['misc'][0]['vals'].keys())

#     print('Plot the Loss vs different param values of Stage 1 optimization')
#     print(optimal_params_keys_list)
#     print('There are only points with loss less this loss quantile:', quantile_of_loss)

    # Number of rows and cols.
    ncols_plot = ncols_plot

    # Number of rows is rounded up
    if (len(optimal_params_keys_list) % ncols_plot) == 0:
        nrows_plot = int(len(optimal_params_keys_list) / ncols_plot)
    else:
        nrows_plot = int(
            (len(optimal_params_keys_list) // ncols_plot) + 1
        )
#     print('cols and rows:', ncols_plot, nrows_plot, end=';   ')

    # Space for each subplot.
    subplot_size_width = 4
    subplot_size_height = 4

    # The size for the whole plot.
    plot_size_inches_width = ncols_plot * subplot_size_width
    plot_size_inches_height = nrows_plot * subplot_size_height

    fig, axes = plt.subplots(nrows=nrows_plot, ncols=ncols_plot)
    fig.set_size_inches(plot_size_inches_width, plot_size_inches_height, forward=True)

    w_list = []

    for key_now in optimal_params_keys_list:

        # Choose subplot.
        key_num = optimal_params_keys_list.index(key_now)
        row_pos = key_num // ncols_plot
        col_pos = key_num % ncols_plot

        # Print the progress of plotting.
    #         print(key_now, key_num, col_pos, row_pos, end=';   ')

        # Plot the loss vs param
        try:
            xs = trails_with_loss_less_than_quantile[key_now].apply(lambda x: round(float(x), 8))
        except:
            xs = trails_with_loss_less_than_quantile[key_now].apply(lambda x: str(x))
        ys = trails_with_loss_less_than_quantile['loss']
        sns.boxplot(  
            x=xs, 
            y=ys, 
            width=0.6,
            color='white',
            orient='v',
            ax=axes[row_pos, col_pos]
        ) ; axes[row_pos, col_pos].set_title(str('param: ' + str(key_now)))
        axes[row_pos, col_pos].set_xticklabels(
            axes[row_pos, col_pos].get_xticklabels(), 
            rotation=90,
        )
        plt.tight_layout()
    
    try:
        axes[2, 0].text(0.01, -0.45, note_text,
            verticalalignment='bottom', horizontalalignment='left',
            transform=axes[2, 0].transAxes,
            color='green', fontsize=12)
    except Exception as e:
        print('Note_text will not be displayed')
        print('Probably because position [2, 0] does not exist')
        print(e)
    
    if save_plot:
        plt.savefig(file_path, bbox_inches = "tight")
        
    if not show_plot:
        plt.close(fig)





def preprocessing_and_hyper_opt_and_plots_gensim_ldamodel(
    texts_df,
    col_name_text_in_df,
    custom_stop_words,
    text_lang='german',
    min_df_cut_off=0.001,
    max_df_cut_off=0.999,
    ngram_range=(1, 1),
    metric_to_optimize='coherence',
    stage_1_params={
        'num_of_topics_min': 10,
        'num_of_topics_max': 100,
        'passes_param': 100,
        'iterations_param': 250,
        'max_eval_param': 5000,
        'timeout_param': 86400,
    },
    stage_2_params={
        'num_of_topics_min': 10,
        'num_of_topics_max': 100,
        'passes_param': 100,
        'iterations_param': 250,
        'max_eval_param': 200,
        'timeout_param': 86400,
    },
    path_to_folder='C:/DAN/t_systems/topic_modelling_project/experiment_feedback/',
    random_seed=420):
    
    file_name_addition = '+col_name_text_' + str(col_name_text_in_df) + '+ngrams_' + str(ngram_range) + '+min_df_' + str(min_df_cut_off) + '+num_topics_min_max_' + str(stage_1_params['num_of_topics_min']) + '_' + str(stage_1_params['num_of_topics_max']) + '+metric_to_optimize_' + str(metric_to_optimize)
    print(file_name_addition)
    
    # Preprocess texts.
    texts_df['text_preprocessed'] = texts_df[col_name_text_in_df].apply(
        lambda x: preprocess_doc(x, lang='german', sw=True, custom_stop_words=custom_stop_words)
    )
    print('texts_df.shape:', texts_df.shape)
    
    # Create document term matrix.
    data_vect, vectorizer, data_dtm = create_document_term_matrix_from_texts_series(
        texts_df['text_preprocessed'],
        ngram_range=ngram_range,
        index_for_dtm=texts_df.index,
        vocabulary_cut_offs={
            'min_df': min_df_cut_off,
            'max_df': max_df_cut_off
        }
    )
    print('data_dtm.shape:', data_dtm.shape)
    
    import scipy
    from scipy.sparse import csr_matrix
    # Create the gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(csr_matrix(data_dtm.transpose()))
    # Create the vocabulary dictionary.
    id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
    
    # Find almost all optimal params of LDA model with fixed number of topics.
    params_optimal_stage_1, trials_stage_1 = stage_1_optimization_gensim_ldamodel(
        corpus=corpus, 
        id2word=id2word,
        metric_to_optimize=metric_to_optimize,
        stage_1_params=stage_1_params,
        random_seed=random_seed
    )
    print('params_optimal_stage_1:',  params_optimal_stage_1)
    
    # Optimize only number of topics with fixed optimal params from Stage 1.
    optimal_params_result, trials_stage_2 = stage_2_optimization_gensim_ldamodel(
        corpus=corpus,
        id2word=id2word,
        params_optimal_stage_1=params_optimal_stage_1,
        metric_to_optimize='coherence',
        stage_2_params=stage_2_params,
        random_seed=random_seed
    )
    print('optimal_params_result:', optimal_params_result)
    
    # Save trials of Stage 1
    trials_stage_1_df = trails_of_hyper_opt_to_dataframe(trials_stage_1)
    trials_stage_1_df.to_csv(path_to_folder + 'trials_stage_1_df' + file_name_addition + '.csv')

    import json
    # Save optimal params from Stage 1.
    with open(path_to_folder + 'params_optimal_stage_1' + file_name_addition + '.json', 'w') as json_file:
        json.dump(params_optimal_stage_1, json_file)
    json_file.close()
    
    # Save trials of Stage 2
    trials_stage_2_df = trails_of_hyper_opt_to_dataframe(trials_stage_2)
    trials_stage_2_df.to_csv(path_to_folder + 'trials_stage_2_df' + file_name_addition + '.csv')

    # Save optimal params for both stages.
    with open(path_to_folder + 'optimal_params_result' + file_name_addition + '.json', 'w') as json_file:
        json.dump(optimal_params_result, json_file)
    json_file.close()

    # Plot the results of Stage 1 optimization
    import matplotlib.pyplot as plt
    
    # Plot the loss vs. params of Stage 1 optimization
    # All trials will be represented
    # Plot will be saved and displayed
    plot_file_name = 'loss_vs_params_all_trials' + file_name_addition + '.png'
    plot_loss_vs_params(
        trials_stage_1_df=trials_stage_1_df,
        quantile_of_loss=1.0,
        ncols_plot=3,
        file_path=path_to_folder+plot_file_name,
        note_text='all_trials_'+file_name_addition,
        save_plot=True,
        show_plot=True
    )

    # Plot the loss vs. params of Stage 1 optimization
    # Only 50% trials witn best loss will be represented
    # Plot will be saved but not displayed
    plot_file_name = 'loss_vs_params_best_half_trials' + file_name_addition + '.png'
    plot_loss_vs_params(
        trials_stage_1_df=trials_stage_1_df,
        quantile_of_loss=0.5,
        ncols_plot=3,
        file_path=path_to_folder+plot_file_name,
        note_text='best_half_'+file_name_addition,
        save_plot=True,
        show_plot=False
    )
    
    # Plot the results of Stage 2 optimization
    plot_file_name = 'loss_vs_num_topics_stage_2_' + file_name_addition + '.png'
    # Show the loss vs. number of topics of Stage 2 optimization.
    f, ax = plt.subplots(1)
    trials_stage_2_df = trials_stage_2_df.sort_values(by='num_topics')
    xs = trials_stage_2_df['num_topics']
    ys = trials_stage_2_df['loss']
    plt.plot(xs, ys, linestyle='solid',color='blue')
    ax.set_title('Loss vs num_topics')
    ax.set_xlabel('num_topics')
    ax.set_ylabel('Loss')
    f.text(0.05, -0.1, str('loss_vs_num_topics_stage_2_' + file_name_addition), 
        fontsize=10, color='green')
    plt.savefig(path_to_folder+plot_file_name, bbox_inches = "tight")
#     plt.close(f)






##############
#            #
# GUIDED LDA #
#            #
##############


# Define the function of single optimization step
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimization_hyper_params_guidedlda(
    data_vect,
    seed_topics,
    dictionary,
    feature_names,
    corpus,
    param_space,
    param_dict={},
    n_iter_param=2000,
    refresh_param=200,
    seed_confidence=0,
    random_seed=420,
    max_eval_param=100,
    timeout_param=600,
    metric_to_optimize='coherence_consistent'):
    
    '''

    '''
    for key in list(param_space.keys()):
        try:
            param_dict.pop(key)
        except:
            pass
    
    def optimize_lda(
        params, param_dict=param_dict,
        data_vect=data_vect,
        seed_topics=seed_topics, seed_confidence=seed_confidence, 
        n_iter=n_iter_param, refresh=refresh_param,
        random_seed=random_seed, 
        corpus=corpus, dictionary=dictionary, feature_names=feature_names,
        metric=metric_to_optimize):
        '''
        Function to minimize in hyper parameter optimization
        metric: 'coherence_consistent', 'loglikelihood'
        '''
        model_guidedlda = guidedlda.GuidedLDA(
            random_state=random_seed,
            n_iter=n_iter, refresh=refresh,
            **params, **param_dict
        )
        model_guidedlda.fit(
            X=data_vect, 
            seed_topics=seed_topics, 
            seed_confidence=seed_confidence
        )

        # to check whether model's params change every trials or not
        # print('alpha,', 'beta,', 'eta,', 'n_topics,', 'random_state')
        # print(model_guidedlda.alpha, model_guidedlda.beta, model_guidedlda.eta, model_guidedlda.n_topics, model_guidedlda.random_state)
        
        if metric=='loglikelihood':
            metric_value = abs(model_guidedlda.loglikelihood())
        elif metric=='coherence_consistent':
            n_top_words = 20
            topic_word = model_guidedlda.topic_word_
            topics_lists = []
            for i, topic_dist in enumerate(topic_word):
                topic_words = list(np.array(feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1])
                topics_lists.append(topic_words)
            coherence_model_lda = gensim.models.CoherenceModel(topics=topics_lists, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence_lda = coherence_model_lda.get_coherence()
            metric_value = abs(coherence_lda)
        else:
            raise ValueError('Choose the metric_to_optimize of the function')
        return metric_value
    
    from functools import partial
    optimization_function = partial(optimize_lda)
    trials = Trials()
    result = hyperopt.fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=max_eval_param,
        timeout=timeout_param, # in seconds
        trials=trials,
    )
    
    return result, trials





alpha_values_list_guidedlda = [ 
    0.50, 0.40, 0.30, 0.20, 0.10, 
    0.08, 0.07, 0.06, 0.05, 
    0.04, 0.03, 0.02, 0.01, 
    0.005, 0.001, 0.0005, 0.0001
]
eta_values_list_guidedlda = [ 
    0.10, 0.09, 0.08, 0.07, 0.06,
    0.05, 0.04, 0.03, 0.02, 0.01, 
    0.008, 0.006, 0.004, 0.002, 0.001, 
    0.0005, 0.0001, 0.00005, 0.00001
]






def stage_1_optimization_guidedlda(
    data_vect,
    seed_topics,
    dictionary,
    feature_names,
    corpus,
    param_dict={},
    metric_to_optimize='coherence_consistent',
    stage_1_params={
        'n_topics_min': 14,
        'n_topics_max': 16,
        'n_iter_param': 2000,
        'refresh_param': 200,
        'seed_confidence': 0,
        'max_eval_param': 1000,
        'timeout_param': 86400, 
    },
    random_seed=420):
    
    # Define some constant values of Stage 1 optimization

    # Values lists for optimization space.
    # alpha_values_list_guidedlda = [ 
    #     0.50, 0.40, 0.30, 0.20, 0.10, 
    #     0.08, 0.07, 0.06, 0.05, 
    #     0.04, 0.03, 0.02, 0.01, 
    #     0.005, 0.001, 0.0005, 0.0001
    # ]
    # eta_values_list_guidedlda = [ 
    #     0.10, 0.09, 0.08, 0.07, 0.06,
    #     0.05, 0.04, 0.03, 0.02, 0.01, 
    #     0.008, 0.006, 0.004, 0.002, 0.001, 
    #     0.0005, 0.0001, 0.00005, 0.00001
    # ]
    
    # Define params space with params lists of values above
    param_space_stage1 = {
        'n_topics': scope.int(
            hp.quniform('n_topics',
                        stage_1_params['n_topics_min'],
                        stage_1_params['n_topics_max'],
                        1)
        ),
        'alpha': hp.choice('alpha', alpha_values_list_guidedlda),
        'eta': hp.choice('eta', eta_values_list_guidedlda),
    }
    
    # Stage 1 optimization with almost all params of the LDA model.
    result_stage_1, trials_stage_1 = optimization_hyper_params_guidedlda(
        data_vect=data_vect,
        seed_topics=seed_topics,
        dictionary=dictionary,
        feature_names=feature_names,
        corpus=corpus,
        param_space=param_space_stage1,
        param_dict=param_dict,
        n_iter_param=stage_1_params['n_iter_param'],
        refresh_param=stage_1_params['refresh_param'],
        seed_confidence=stage_1_params['seed_confidence'],
        random_seed=random_seed,
        max_eval_param=stage_1_params['max_eval_param'],
        timeout_param=stage_1_params['timeout_param'],
        metric_to_optimize=metric_to_optimize
    )
    
    # Params from optimization stage 1 without num_topics. Fix them for Stage 2.
    params_optimal_stage_1 = {
        'alpha': alpha_values_list_guidedlda[result_stage_1['alpha']],
        'eta': eta_values_list_guidedlda[result_stage_1['eta']],
    }
    
    return params_optimal_stage_1, trials_stage_1








def stage_2_optimization_guidedlda(
    data_vect,
    seed_topics,
    dictionary,
    feature_names,
    corpus,
    param_dict={}, # add **params_optimal_stage_1 in deploy
    metric_to_optimize='coherence_consistent',
    stage_2_params={
        'n_topics_min': 14,
        'n_topics_max': 16,
        'n_iter_param': 2000,
        'refresh_param': 200,
        'seed_confidence': 0,
        'max_eval_param': 1000,
        'timeout_param': 86400, 
    },
    random_seed=420):
    
    '''
    This function 
    '''
    
    # Define some values of Stage 2 optimization

    # Param space for Stage 2.
    param_space_stage2 = {
        'n_topics': scope.int(
            hp.quniform('n_topics',
                        stage_2_params['n_topics_min'],
                        stage_2_params['n_topics_max'],
                        1)
        ),
    }
    
    # Stage 2 optimization with fixed optimal params from Stage 1.
    result_stage_2, trials_stage_2 = optimization_hyper_params_guidedlda(
        data_vect=data_vect,
        seed_topics=seed_topics,
        dictionary=dictionary,
        feature_names=feature_names,
        corpus=corpus,
        param_space=param_space_stage2,
        param_dict=param_dict,
        n_iter_param=stage_2_params['n_iter_param'],
        refresh_param=stage_2_params['refresh_param'],
        seed_confidence=stage_2_params['seed_confidence'],
        random_seed=random_seed,
        max_eval_param=stage_2_params['max_eval_param'],
        timeout_param=stage_2_params['timeout_param'],
        metric_to_optimize=metric_to_optimize
    )
    
    # Define optimal params values as joined dictionary from both Stages
    optimal_params_result = {**param_dict, **result_stage_2}
    
    return optimal_params_result, trials_stage_2








# Define a function for two stage optimization of LDA topic modelling.
# Stage 1 - optimize params of LDA model with fixed number of topics.
# Then take these optimal params of Stage 1 and fix them for Stage 2.
# Stage 2 - optimize number of topics with fixed other params of LDA model.
def two_stage_optimization_guidedlda(
    data_vect,
    seed_topics,
    dictionary,
    feature_names,
    corpus,
    metric_to_optimize='coherence_consistent',
    stage_1_params={
        'n_topics_min': 18,
        'n_topics_max': 21,
        'n_iter_param': 2000,
        'refresh_param': 200,
        'seed_confidence': 0,
        'max_eval_param': 1000,
        'timeout_param': 86400, 
    },
    stage_2_params={
        'n_topics_min': 10,
        'n_topics_max': 100,
        'n_iter_param': 2000,
        'refresh_param': 200,
        'seed_confidence': 0,
        'max_eval_param': 1000,
        'timeout_param': 86400, 
    },
    random_seed=420):
    
    # Stage 1 optimization with almost all params of the LDA model.
    params_optimal_stage_1, trials_stage_1 = stage_1_optimization_guidedlda(
        data_vect=data_vect,
        seed_topics=seed_topics,
        dictionary=dictionary,
        feature_names=feature_names,
        corpus=corpus,
        param_dict={},
        metric_to_optimize='coherence_consistent',
        stage_1_params=stage_1_params,
        random_seed=420
    )
    
    # Stage 2 optimization with fixed optimal params from Stage 1.
    optimal_params_result, trials_stage_2 = stage_2_optimization_guidedlda(
        data_vect=data_vect,
        seed_topics=seed_topics,
        dictionary=dictionary,
        feature_names=feature_names,
        corpus=corpus,
        param_dict=params_optimal_stage_1, # add **params_optimal_stage_1 in deploy
        metric_to_optimize='coherence_consistent',
        stage_2_params=stage_2_params,
        random_seed=420
    )
    
    return optimal_params_result, trials_stage_1, trials_stage_2






import sys
def trails_of_hyper_opt_to_dataframe_guidedlda(trials_object):
    '''
    This function takes hyperopt object with trials
    and converts this object to dataframe.

    It adds columns with params values and function loss values for each trial.
    '''
    # Convert list of dictionaries to df
    df_with_trials = pd.DataFrame(trials_object.trials)

    # But we need to make columns of params and results from dictionaries.
    # Take param names of trials and possible result outputs
    list_of_params_in_trials = list(df_with_trials['misc'][0]['vals'].keys())
    list_of_result_in_trials = list(df_with_trials['result'][0].keys())

    # Iterate these two dictionaries to create columns
    for param_now in list_of_params_in_trials:
        df_with_trials[param_now] = df_with_trials['misc'].apply(
            lambda x: x['vals'][param_now][0]
        )
    for result_now in list_of_result_in_trials:
        df_with_trials[result_now] = df_with_trials['result'].apply(
            lambda x: x[result_now]
        )
    # Iterate only columns with params to convert thier indexes to values.
    for param_now in list_of_params_in_trials:
        try:
            # num_topics param is not index,
            # so converting to values is not needed.
            if param_now == 'num_topics':
                continue
            # Param space for Stage 1 was defined as combination of values lists.
            # These lists were defined above (e.g. alpha_values_list).
            # So values_list variable is defined as one of these lists.
            values_list = getattr(sys.modules[__name__], str(str(param_now) + '_values_list_guidedlda'))
            # Convert indexes of params columns to their values
            # by replacing index with one of the value of list with this index.
            df_with_trials[param_now] = df_with_trials[param_now].apply(
                lambda x: values_list[x]
            )
        except Exception as e:
            print(e)
    
    return df_with_trials