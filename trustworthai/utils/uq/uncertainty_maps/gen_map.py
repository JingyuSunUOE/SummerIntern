import torch
import numpy as np

"""
'However, variance is not sufficiently representative in the con-
text of multi-modal distributions.' - https://arxiv.org/pdf/1807.07356.pdf
chpt 3
"""

def samples_entropy_map(samples, 
                        p_from_mode=True,
                        threshold_preds=False,
                        preds_threshold=0.9,
                        apply_norm=False,
                        norm_func='sigmoid',
                        apply_binning=False,
                        binning_dp = 1,
                        supress_warning=False,
                        print_uniques=False,
                       ):
    """
    H(y|x) = - sum_1^M{p_hat_i_m ln(p_hat_i_m)}
    where p_hat_m is the frequency of the m-th unique value.
    
    Okay so this implies that my model results have been turned directly
    into classes which currently my model outputs are not
    
    so should I take the mode or the mean..?
    
    samples: the different samples taken for the input batch
    p_from_mode: the different smaples taken for the input batch
    """
    # i.e should be number of classes, but could also represent
    # probability of being class 1, so I need to decide what I am rrepresenting
    # the former makes this function easier and fits mroe nicely with the 'semantic segmentation idea'.
    # but ideally the model outputs a probability... hmm should clear this up with them.
    
    if threshold_preds:
        if not supress_warning:
            print("warning, thresholding only valid for binary class problem")
        samples = samples > preds_threshold
        
    if apply_binning:
        if not supress_warning:
            print("warning, binning should only be applied where output is a probability")
        samples = np.around(samples, binning_dp)
    
    return entropy(samples, supress_warning, print_uniques)
    

def entropy(samples, supress_warning=False, print_uniques=False):
    """
    calculates entropy where samples is in shape
    [s, b, x, y, (z)] s = samples, b = batch size, 
    note there should not be multiple channels
    """
    if not supress_warning:
        print("warning: multiclass problems should be presented as single channel, value = id of class")
    samples = np.array(samples)
    V = len(samples)
    samples = np.moveaxis(samples, 0, -1)
    # print(samples.shape)
    uniques = np.unique(samples)
    if print_uniques:
        print(uniques)
    entropy = np.zeros(samples.shape[:-1])
    for u in uniques:
        p_m_hat = np.sum([samples==u], axis=-1).squeeze() / V
        entropy += - p_m_hat * np.ma.log(p_m_hat).filled(0) # where p doesnt occur just fill with zeros.
    return entropy
    

def samples_variance_map(samples):
    """
    calculates entropy where samples is in shape
    [s, b, x, y, (z)] s = samples, b = batch size, 
    note there should not be multiple channels
    """
    return np.var(samples, axis=0)

