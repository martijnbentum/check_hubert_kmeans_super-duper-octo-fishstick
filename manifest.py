import locations
from pathlib import Path

def load_manifest(filename = None):
    if not filename:
        filename = locations.pretrain_manifest
    filename = Path(filename)
    print(f'loading manifest file {filename}')
    with filename.open() as fin:
        t = [x.split('\t') for x in fin.read().split('\n') if x]
    header, data = t[0], t[1:]
    data = [[x[0], int(x[1])] for x in data]
    return header, data

def filename_to_corpus(filename):
    d = {'cgn':'cgn_','mls':'MLS','cv':'common_voice'}
    for corpus, check in d.items():
        if check in filename: return corpus
    raise ValueError(f'could not determine corpus of file {filename}')


def select_corpus(corpus = 'cgn', manifest_data = None, 
    manifest_filename = None):
    if corpus not in ['cgn','mls','cv']: 
        raise NotImplemented(f'{corpus} not available')
    if manifest_data is None:
        _, manifest_data = load_manifest(manifest_filename)
    output = []
    for line in manifest_data:
        if filename_to_corpus(line[0]) == corpus: output.append(line)
    return output

def select_cgn(manifest_data = None, manifest_filename= None):
    return select_corpus('cgn',manifest_data, manifest_filename)

def select_mls(manifest_data = None, manifest_filename= None):
    return select_corpus('mls',manifest_data, manifest_filename)


def select_cv(manifest_data = None, manifest_filename= None):
    return select_corpus('cv',manifest_data, manifest_filename)

def split_manifest_in_corpora(manifest_data = None, manifest_filename = None):
    if manifest_data is None:
        _, manifest_data = load_manifest(manifest_filename)
    d = {}
    d['cgn'] = select_cgn(manifest_data)
    d['mls'] = select_mls(manifest_data)
    d['cv'] = select_cv(manifest_data)
    return d

def manifest_line_to_number_of_frames(line, audio_sample_rate = 16000,
    feature_sample_rate = 100):
    samples_per_frame = audio_sample_rate / feature_sample_rate
    samples = line[-1]
    frames = samples / samples_per_frame 
    duration = samples / audio_sample_rate
    return frames, duration
    

