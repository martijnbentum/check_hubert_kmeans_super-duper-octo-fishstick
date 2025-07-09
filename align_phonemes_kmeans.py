import locations
import kmeans
from pathlib import Path
import phonemes
import pnmi
from progressbar import progressbar
import numpy as np



def handle_frames(frames):
    for f in frames.frames:
        f.phoneme = None
    for phoneme in frames.phoneme_labels:
        st = float(phoneme['start_time'])
        et = float(phoneme['end_time'])
        fs = frames.select_frames(st, et, 
            percentage_overlap = 50)
        for f in fs:
            f.phoneme = phoneme['phoneme']
    frames.phonemes = []
    frames.phone_kmeans = []
    for f in frames.frames:
        frames.phonemes.append(f.phoneme)
        if f.phoneme:
            frames.phone_kmeans.append((f.phoneme, f.label()))

def handle_corpus(corpus = 'cgn'):
    d = phonemes.make_filename_to_phoneme_labels_dict(corpus)
    label_filenames = locations.corpus_to_kmeans_label_filenames()[corpus]
    output = []
    phone_labels, kmean_labels= [], []
    bads = []
    part = []
    pnmis = []
    for i,filename in progressbar(enumerate(label_filenames)):
        frames = kmeans.label_filename_to_frame_labels(filename)
        if filename.stem not in d.keys():
            bads.append(filename)
            continue
        pl = d[filename.stem] 
        frames.phoneme_labels = pl
        handle_frames(frames)
        output.append(frames)
        p, k = list(zip(*frames.phone_kmeans))
        phone_labels.extend(p)
        kmean_labels.extend(k)
        part.extend(frames.phone_kmeans)
        if i > 0 and i % int(len(label_filenames) / 10) == 0:
            p, k = list(zip(*part))
            pnmis.append(  pnmi.pnmi(p,k) )
            part = []
    print('pnmi:', pnmi.pnmi(phone_labels, kmean_labels))
    print('pnmi mean / std:', np.mean(pnmis), np.std(pnmis))
    print(f'aligned {len(output)} files, missed {len(bads)}')
    return output, pnmis

def handle_cgn():
    return handle_corpus('cgn')

def handle_mls():
    return handle_corpus('mls')

def handle_cv():
    return handle_corpus('cv')
    

