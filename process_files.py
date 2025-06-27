import kmeans
import locations
from pathlib import Path
from progressbar import progressbar
import random

def process_cgn(n_files = 100, model = None, model_filename = None):
    if model is None: model = kmeans.load_kmeans_model(model_filename)
    fn = list(locations.cgn_news_sentences.glob('*.wav'))
    for f in progressbar(fn[:n_files]):
        output_filename = locations.kmeans_labels / (f.stem + '.txt')
        if output_filename.exists(): continue
        labels = kmeans.audio_filename_to_labels(f, model)
        kmeans.save_labels(labels, output_filename)

def process_mls(n_files = 100, model = None, model_filename = None):
    if model is None: model = kmeans.load_kmeans_model(model_filename)
    fn = get_mls_filenames(n_files)
    for f in progressbar(fn[:n_files]):
        f = Path(f)
        output_filename = locations.kmeans_labels / (f.stem + '.txt')
        if output_filename.exists(): continue
        labels = kmeans.audio_filename_to_labels(f, model)
        kmeans.save_labels(labels, output_filename)

def process_cv(n_files = 100, model = None, model_filename = None):
    if model is None: model = kmeans.load_kmeans_model(model_filename)
    fn = get_cv_filenames(n_files)
    for f in progressbar(fn[:n_files]):
        f = Path(f)
        output_filename = locations.kmeans_labels / (f.stem + '.txt')
        if output_filename.exists(): continue
        labels = kmeans.audio_filename_to_labels(f, model)
        kmeans.save_labels(labels, output_filename)

def load_manifest():
    with open(locations.pretrain_manifest) as fin:
        manifest = fin.read().split('\n')
    return manifest[1:-1]

def get_mls_filenames(n_files = 100):
    random.seed(9)
    manifest = load_manifest()
    random.shuffle(manifest)
    output = [] 
    directory = locations.pretrain_datasets
    for line in manifest:
        if not 'MLS' in line: continue
        f = directory / line.split('\t')[0]
        if not f.exists():
            raise ValueError(f'could not find {f}')
        output.append(str(f))
        if len(output) >= n_files:break
    return output

    
def get_cv_filenames(n_files = 100):
    random.seed(9)
    manifest = load_manifest()
    random.shuffle(manifest)
    output = [] 
    directory = locations.pretrain_datasets
    for line in manifest:
        if not 'common_voice' in line: continue
        f = directory / line.split('\t')[0]
        if not f.exists():
            raise ValueError(f'could not find {f}')
        output.append(str(f))
        if len(output) >= n_files:break
    return output

