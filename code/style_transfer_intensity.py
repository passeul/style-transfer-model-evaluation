"""EVALUATION OF STYLE TRANSFER INTENSITY

This code can be used for evaluation of style transfer intensity (STI) between input and output sentiment texts of a style transfer model.

Past evaluations have used the number of output texts with the desired target style.
Note that STI, however, makes use of both the input and the output texts for a more meaningful evaluation.
STI captures more information, i.e. how *much* the style changed from the input to the output.

Two output texts could exhibit the same overall target style, but with one being more pronounced than the other,
e.g. "i like this" vs. "i love this !" While past evaluations do not quantify that, STI can, 
given style distributions from a style classifier trained on labeled style datasets.

The following style classifiers were trained on ../data/sentiment.train.* and 
used in experiments to obtain style distributions:
    - fasttext (https://github.com/facebookresearch/fastText)
    - textcnn (https://github.com/DongjunLee/text-cnn-tensorflow)

STI, based on Earth Mover's Distance (EMD) between style distributions of input and output texts,
exhibited higher correlation with human evaluations of the same texts than did the method of past evaluations
(using target style scores of output texts alone).

Usage:
    - View correlations between automated metrics and human scores                  -> display_correlation_tables()
    - Load style distributions for output texts of examined style transfer models   -> load_style_distributions(...)
    - Calculate STI for your own input/output text style distributions              -> calculate_direction_corrected_emd(...)
   
"""

from globals import MODEL_TO_PARAMS, MODEL_TO_PARAM_NAMES
from pyemd import emd
from utils import calculate_correlations, get_val_as_str, load_turk_scores
import numpy as np

ASPECT = 'style_transfer_intensity'
AUTOMATED_SCORES_PATH = f'../evaluations/automated/{ASPECT}/sentence_level/scores_based_on_emd'
STYLE_DISTRIBUTIONS_PATH = f'../evaluations/automated/{ASPECT}/sentence_level/style_distributions'


## SCORING OF DIRECTION-CORRECTED EMD
def load_style_distributions(path): 
    distributions = np.load(path)
    return np.array(distributions['fasttext']).astype('float64'), np.array(distributions['textcnn']).astype('float64')

def calculate_emd(input_distribution, output_distribution):   
    '''
    Calculate Earth Mover's Distance (aka Wasserstein distance) between 
    two distributions of equal length.

    Parameters
    ----------
    input_distribution : numpy.ndarray
        Probabilities assigned to style classes for an input text
    output_distribution : numpy.ndarray
        Probabilities assigned to style classes for an output text, e.g. of a style transfer model
        
    Returns
    -------
    Earth Mover's Distance (float) between the two given style distributions

    '''
    
    N = len(input_distribution)
    distance_matrix = np.ones((N, N))
    return emd(input_distribution, output_distribution, distance_matrix)

def account_for_direction(input_target_style_probability, output_target_style_probability):
    '''
    In the context of EMD, more mass (higher probability) placed on a target style class
    in the style distribution of an output text (relative to that of the input text)
    indicates movement in the correct direction of style transfer. 
    
    Otherwise, the style transfer intensity score should be penalized, via application
    of a negative direction factor.

    Parameters
    ----------
    input_target_style_probability : float
        Probability assigned to target style in the style distribution of an input text
    output_target_style_probability : float
        Probability assigned to target style in the style distribution of an output text, e.g. of a style transfer model
        
    Returns
    -------
    1 if correct direction of style transfer, else -1

    '''
    
    if output_target_style_probability >= input_target_style_probability:
        return 1
    return -1

def calculate_direction_corrected_emd(input_distribution, output_distribution, target_style_class): 
    '''
    Calculate Earth Mover's Distance (aka Wasserstein distance) between 
    two distributions of equal length, with correction for direction.
    That is, penalize the score if the output style distribution displays
    change of style in the wrong direction, i.e. away from the target style.

    Parameters
    ----------
    input_distribution : numpy.ndarray
        Probabilities assigned to style classes for an input text
    output_distribution : numpy.ndarray
        Probabilities assigned to style classes for an output text, e.g. of a style transfer model
    target_style_class : int
        Label of the intended style class for a style transfer task
        
    Returns
    -------
    Direction-corrected Earth Mover's Distance (float) between the two given style distributions

    '''
    
    emd_score = calculate_emd(input_distribution, output_distribution)
    direction_factor = account_for_direction(input_distribution[target_style_class], output_distribution[target_style_class])
    return emd_score*direction_factor


## CALCULATION OF CORRELATIONS
def extract_scores_for_style_class(style_distributions, style_class):
    '''
    Given style distributions for a set of texts,
    extract probabilities for a given style class
    across all texts.
    
    Parameters
    ----------
    style_distributions : numpy.ndarray
        Style distributions for a set of texts
    style_class : int
        Number representing a particular style in a set of styles, 
        e.g. 0 for negative sentiment and 1 for positive sentiment
        
    Returns
    -------
    Probabilities (numpy.ndarray) for the given style class across all texts

    '''
    
    return style_distributions[:,style_class]

def display_correlation_tables():
    '''
    Display correlation of automated style transfer metrics with
    averaged human evaluations of outputs of examined style transfer models.
    
    ''' 
    
    # load style distributions for input texts
    input_text_fasttext_distr, input_text_textcnn_distr = load_style_distributions(f'{STYLE_DISTRIBUTIONS_PATH}/input_distributions.npz')
            
    # labels used in experiments
    negative_style_class = 0
    positive_style_class = 1
    
    for model in MODEL_TO_PARAMS:
        print(model)
        param_name = MODEL_TO_PARAM_NAMES[model]
        param_values = MODEL_TO_PARAMS[model]

        automated_scores_over_model_params = {}
        turk_scores_over_model_params = []

        for val in param_values:
            string_val = get_val_as_str(val)
            path = f'{STYLE_DISTRIBUTIONS_PATH}/{model}_{MODEL_TO_PARAM_NAMES[model]}_{string_val}.npz'
            
            # load style distributions for output text of style transfer model
            output_text_fasttext_distr, output_text_textcnn_distr = load_style_distributions(path)
            
            # load human evaluations for output texts of style transfer model
            turk_scores = load_turk_scores(ASPECT, model, param_name, string_val)
            turk_scores_over_model_params.extend(turk_scores)
            number_of_scores = len(turk_scores)

            # first half of texts were used in a style transfer task of - to + sentiment (so target style class = 1)
            # second half, + to - (target style class = 0)
            split_point = int(number_of_scores/2)
            
            
            # collect target style scores
            fasttext_target_style_scores = np.concatenate([
                    extract_scores_for_style_class(output_text_fasttext_distr, positive_style_class)[:split_point],
                    extract_scores_for_style_class(output_text_fasttext_distr, negative_style_class)[split_point:]
            ])
            
            textcnn_target_style_scores = np.concatenate([
                    extract_scores_for_style_class(output_text_textcnn_distr, positive_style_class)[:split_point],
                    extract_scores_for_style_class(output_text_textcnn_distr, negative_style_class)[split_point:]
            ])
            
            automated_scores_over_model_params.setdefault('fasttext (target style scores)', []).extend(fasttext_target_style_scores)
            automated_scores_over_model_params.setdefault('textcnn  (target style scores)', []).extend(textcnn_target_style_scores)
            
            
            # collect style transfer intensities based on EMD
            fasttext_intensities = []
            textcnn_intensities = []

            for i in range(number_of_scores):
                # see note about purpose of split_point above
                target_style_class = int(i < split_point)

                # if output does not show greater probability than input for target style class, negate the EMD score
                fasttext_intensity = calculate_direction_corrected_emd(input_text_fasttext_distr[i], output_text_fasttext_distr[i], target_style_class)
                textcnn_intensity = calculate_direction_corrected_emd(input_text_textcnn_distr[i], output_text_textcnn_distr[i], target_style_class)

                fasttext_intensities.append(fasttext_intensity)
                textcnn_intensities.append(textcnn_intensity)
            
            automated_scores_over_model_params.setdefault('fasttext (EMD-based)', []).extend(fasttext_intensities)
            automated_scores_over_model_params.setdefault('textcnn  (EMD-based)', []).extend(textcnn_intensities)
        
        correlation_tables = calculate_correlations(automated_scores_over_model_params, turk_scores_over_model_params)
        print(correlation_tables.round(decimals=3))
        print()