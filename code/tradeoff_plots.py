"""ASPECT TRADEOFF PLOTS

This code can be used to construct tradeoff plots between aspects of interest for a more holistic view of style transfer model performance, as well as for a direct comparison of multiple models.

Each point on a plot represents a corpus-level score achieved with a different hyperparameter setting during model training. The score comes from the automated metric we empirically identified as most strongly correlated or in agreement with human judgments for a given aspect of interest, namely:
    - direction-corrected Earth Mover's Distance (style transfer intensity)
    - Word Mover's Distance (content preservation)
    - scores of output texts from adversarial classifiers (naturalness)
    
With points of intersection (or near intersection), one can identify where different models achieve similar results for various aspects. This can also be useful for understanding the impact of decisions made during model design and optimization phases.
    
Plots will naturally vary by dataset, especially because the tradeoff between content preservation and style transfer intensity 
depends on the level of distinction between style words and content words of the given dataset.

Usage:
    - Load aspect evaluation scores for a given style transfer model    -> TradeoffPlotParameters(...)
    - Plot aspect tradeoffs using the loaded scores                     -> plot_aspect_tradeoffs(...)
    
You can find examples of more detailed usage commands below.  

"""

from globals import *
from matplotlib import pyplot as plt, lines as mlines
from numpy.polynomial import Polynomial
import numpy as np


## DATA LOADING
def load_corpus_level_scores(model, aspect): 
    # To see how these scores were calculated, refer to calculate_corpus_level_scores() in utils.py
    scores_path = f'../evaluations/automated/{aspect}/corpus_level/{model}.npz'
    return np.load(scores_path)[PREFERRED_AUTOMATED_METRICS[aspect]].item()

def normalize_inverses(values):
    inverse_values = np.reciprocal(values)
    return inverse_values / np.linalg.norm(inverse_values)

class TradeoffPlotParameters(object):
    '''Object with aspect evaluation scores for a given style transfer model and plot metadata that can be used to populate a tradeoff plot.'''
    
    def __init__(self, model_name, color, line_style, marker, marker_size=5):
        '''
        Parameters
        ----------
        model_name : str
            One of three style transfer models used in experiments (see globals.py) 
        color : any matplotlib color
            Color of the tradeoff plot
        line_style : matplotlib.lines.Line2D.lineStyles
            Linestyle of the tradeoff plot
        marker : matplotlib.markers
            Line marker
        marker_size : float
            Size of marker
            
        '''
        
        self.model_name = model_name
        self.parameters = MODEL_TO_PARAMS[model_name]
        
        # corpus-level scores from automated metrics
        self.content_preservation_scores = load_corpus_level_scores(model_name, 'content_preservation')
        self.style_transfer_intensities = load_corpus_level_scores(model_name, 'style_transfer_intensity')
        self.naturalness_scores = load_corpus_level_scores(model_name, 'naturalness')
        
        # see https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html for plot marker options  
        self.color = color
        self.legend_title = model_name
        self.line_style = line_style
        self.marker = marker
        self.marker_size = marker_size
    
    def get_aspect_values(self, aspect):
        if aspect == 'content_preservation':
            parameters_to_values = self.content_preservation_scores
        elif aspect == 'style_transfer_intensity':
            parameters_to_values = self.style_transfer_intensities
        elif aspect == 'naturalness':
            parameters_to_values = self.naturalness_scores
        else: 
            raise ValueError(f'Aspect "{aspect}" was not found!')
            
        # extract aspect scores using order of model hyperparameter values
        values = [parameters_to_values[key] for key in sorted(parameters_to_values.keys())]
        
        # WMD is lower when texts are more similar
        # take normalized inverses of scores for the true degree of content preservation
        if aspect == 'content_preservation':
            values = normalize_inverses(values)
            
        return values
    
    
## GENERATION OF TRADEOFF PLOTS
def plot_best_fit_curve(x, y, color, line_style, order):
    p = Polynomial.fit(x, y, order)
    handler, = plt.plot(*p.linspace(n=100, domain=(0,1)), color=color, linestyle=line_style)
    return handler

def plot_aspect_tradeoffs(model_data, aspect1, aspect2, axis_labels, dpi=600, plot_range=(0,1), use_color=True):
    '''
    Generate and save an aspect tradeoff plot for two given aspects of interest. 

    Parameters
    ----------
    model_data : list
        List of TradeoffPlotParameters (one per style transfer model)
    aspect1 : str
        One of three key aspects of style transfer model evaluation (see globals.py)
    aspect2 : str
        One of three key aspects of style transfer model evaluation (see globals.py)
    axis_labels : dict
        Mapping from aspect name to desired text label for the axis which represents it
    dpi : int
        Resolution in dots per inch to use when saving plot
    plot_range : tuple
        Tuple with start and end points for axes
    use_color : bool
        Set to False to generate plot in black and white

    '''  
    
    if len(model_data) == 0:
        raise ValueError("Please enter a list of at least one valid TradeoffPlotParameters object.")
    
    if aspect1 == aspect2:
        raise ValueError("Aspects 1 and 2 cannot be the same!")
        
    # comment out the following line to use default matplotlib font 
    plt.rc('font', family='FreeSerif')
    
    legend_markers = []

    for md in model_data:        
        aspect1_values = md.get_aspect_values(aspect1)
        aspect2_values = md.get_aspect_values(aspect2)    
        aspect_pair_values = list(zip(aspect1_values, aspect2_values))

        color = md.color if use_color else 'black' 

        # plot points
        for x,y in aspect_pair_values:
            plt.plot(x, y, color='black', label=md.legend_title, marker=md.marker, markersize=md.marker_size)

        # plot best fit line 
        legend_handler = plot_best_fit_curve(aspect1_values, aspect2_values, color=color, line_style=md.line_style, order=1)
        
        # collect legend marker
        legend_marker = mlines.Line2D([], [], color=color, label=md.legend_title, linestyle=md.line_style, marker=md.marker, markersize=md.marker_size)
        legend_markers.append(legend_marker)
    
    plt.legend(handles=legend_markers)
    
    plt.xlabel(axis_labels[aspect1])
    plt.ylabel(axis_labels[aspect2])
        
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    
    figure_path = f'../evaluations/automated/tradeoff_plots/{aspect2}_vs_{aspect1}.png'
    plt.savefig(figure_path, dpi=dpi)   
    
    plt.show()
    plt.clf()
    
    
## EXAMPLE USAGE (uncomment the following to generate aspect tradeoff plots)

# arae_plot_parameters = TradeoffPlotParameters('ARAE', 'blue',   ':', 's')
# caae_plot_parameters = TradeoffPlotParameters('CAAE', 'red', '--', 'o')
# dar_plot_parameters  = TradeoffPlotParameters('DAR',  'green', '-', 'D')
# all_model_parameters = [arae_plot_parameters, caae_plot_parameters, dar_plot_parameters]

# axis_labels = {
#     'style_transfer_intensity': f'style transfer intensity ({PREFERRED_AUTOMATED_METRICS["style_transfer_intensity"]} EMD-based)',
#     'content_preservation': 'content preservation (normalized inverse of WMD)',
#     'naturalness': f'naturalness (scores of {PREFERRED_AUTOMATED_METRICS["naturalness"]} classifier)'
# }

# plot_aspect_tradeoffs(all_model_parameters, 'style_transfer_intensity', 'content_preservation', axis_labels)
# plot_aspect_tradeoffs(all_model_parameters, 'style_transfer_intensity', 'naturalness', axis_labels)