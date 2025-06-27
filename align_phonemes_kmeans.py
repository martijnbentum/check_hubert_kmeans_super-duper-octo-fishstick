import locations
import kmeans
from pathlib import Path
import phonemes
from progressbar import progressbar


def handle_corpus(corpus = 'cgn'):
    d = phonemes.make_filename_to_phoneme_labels_dict(corpus)
    label_filenames = locations.corpus_to_kmeans_label_filenames()[corpus]
    output = []
    for filename in progressbar(label_filenames):
        print(filename)
        frames = kmeans.label_filename_to_frame_labels(filename)
        phoneme_labels = d[filename.stem] 
        frames.phoneme_labels = phoneme_labels
        output.append(frames)
    return output

def handle_cgn():
    return handle_corpus('cgn')

def handle_mls():
    return handle_corpus('mls')

def handle_cv():
    return handle_corpus('cv')
    

