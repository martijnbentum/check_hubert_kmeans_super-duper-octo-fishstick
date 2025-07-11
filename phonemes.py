import json
import locations
from pathlib import Path
from progressbar import progressbar


def make_filename_to_phoneme_labels_dict(corpus = 'cgn',header = None, 
    data = None, output_filename = None):
    if not output_filename:
        output_filename = f'../{corpus}_filename_to_phoneme_labels.json'
    if Path(output_filename).exists():
        with open(output_filename) as fin:
            d = json.load(fin)
        return d
    if not header or not data:
        if corpus == 'cgn': f = open_cgn_news
        elif corpus == 'cv': f = open_cv
        elif corpus == 'mls': f = open_mls
        else: raise ValueError('unknown corpus', corpus)
        header, data = f()
    label_filenames = locations.corpus_to_kmeans_label_filenames()[corpus]
    d ={}
    bads = []
    errors = []
    for filename in progressbar(label_filenames):
        try:labels=filename_to_phoneme_labels(filename, header, data)
        except ValueError as e:
            print(f'{e} should be FA error or no phone labels')
            errors.append(filename)
        else: 
            if len(labels) > 0:
                d[filename.stem] = labels
            else: bads.append(filename)
    with open(output_filename, 'w') as fout:
        json.dump(d, fout)
    return d, bads, errors



def filename_to_phoneme_labels(filename, header, data):
    output = []
    filename_index = header.index('audio_filename')
    identifier = Path(filename).stem
    start_found = False
    for line in data:
        f = line[filename_index]
        if identifier == f.split('.')[0]: 
            pd = line_to_phoneme_dict(line, header)
            phoneme = pd['phoneme']
            if '!' in phoneme:
                raise ValueError(f'{phoneme} contains "!" -> bad FA')
            output.append(pd)
            start_found = True
        elif start_found: break
    if len(output) == 0:
        raise ValueError(f'{filename}, no phone labels')
    return output


def line_to_phoneme_dict(line, header):
    if 'phoneme' in header: phoneme_index = header.index('phoneme')
    elif 'phone' in header: phoneme_index = header.index('phone')
    else: raise ValueError('no column named phone or phoneme')
    d = {}
    d['filename'] = line[header.index('audio_filename')].split('.')[0]
    d['phoneme'] = line[phoneme_index]
    d['start_time'] = line[header.index('start_time')]
    d['end_time'] = line[header.index('end_time')]
    return d
    

def open_cgn_news():
    with locations.cgn_news_phonemes.open() as fin:
        t = [line.split('\t') for line in fin.read().split('\n')]
    header = t[0]
    data = t[1:]
    return header, data

def open_mls():
    with locations.mls_phonemes.open() as fin:
        t = [line.split('\t') for line in fin.read().split('\n')]
    header = t[0]
    data = t[1:]
    return header, data

def open_cv():
    with locations.cv_phonemes.open() as fin:
        t = [line.split('\t') for line in fin.read().split('\n')]
    header = t[0]
    data = t[1:]
    return header, data

