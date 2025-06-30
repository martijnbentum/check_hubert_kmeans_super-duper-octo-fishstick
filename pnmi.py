from collections import Counter
import math
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

def entropy(phoneme_labels):
    '''Compute entropy H(y) in bits.
    '''
    labels = np.asarray(phoneme_labels)
    n = len(labels)
    counts = Counter(phoneme_labels)
    
    h = 0.0
    for count in counts.values():
        p = count / n
        h -= p * math.log2(p)
    
    return h

def conditional_entropy(phoneme_labels, kmean_labels):
    '''Compute H(y|z): conditional entropy in bits.
    '''
    y = np.asarray(phoneme_labels)
    z = np.asarray(kmean_labels)
    n = len(y)

    # Group y's by z's
    joint_counts = Counter(zip(y, z))
    z_counts = Counter(z)

    ce = 0.0
    for (y_val, z_val), joint_count in joint_counts.items():
        p_yz = joint_count / n
        p_z = z_counts[z_val] / n
        ce += p_yz * math.log2(p_z / p_yz)  
    return ce

def pnmi(phoneme_labels, kmean_labels):
    '''compute the phone normalized mutual information
    1 - conditional entropy of phone labels given kmean labels divided by the
    entropy of the phone labels
    1 - H(phone labels | kmean labels) / H(phone labels)
    '''
    H = entropy(phoneme_labels)
    ce = conditional_entropy(phoneme_labels, kmean_labels)
    return 1 - (ce / H)

def pnmi_check(phoneme_labels, kmean_labels):
    '''this should equal the pnmi function
    mutual information of phone labels ; kmean labels divided by the entropy
    of the phone labels
    mi(phone labels ; kmean labels) / H(phone labels)
    '''
    H = entropy(phoneme_labels)
    mi = mutual_info_score(phoneme_labels, kmean_labels)
    mi /= math.log(2) # nats to bits
    return mi / H


