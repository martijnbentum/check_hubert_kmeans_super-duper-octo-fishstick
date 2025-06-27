from collections import Counter
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

def entropy_from_labels(labels):
    _, counts = np.unique(labels, return_counts = True)
    return entropy(counts / counts.sum(), base = 2) 

def pmni(phoneme_labels, kmeans_labels):
    mi = mutual_info_score(phoneme_labels, kmeans_labels)
    H = entropy_from_labels(phoneme_labels)
    return 1 - ( mi / H )
