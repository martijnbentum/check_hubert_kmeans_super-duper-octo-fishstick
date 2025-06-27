import kmeans
import locations
from pathlib import Path
from progressbar import progressbar

def process_cgn(n_files = 1000, model = None, model_filename = None):
    if model is None: model = kmeans.load_kmeans_model(model_filename)
    fn = list(locations.cgn_news_sentences.glob('*.wav'))
    for f in progressbar(fn[:n_files]):
        output_filename = locations.kmeans_labels / (f.stem + '.txt')
        if output_filename.exists(): continue
        labels = kmeans.audio_filename_to_labels(f, model)
        kmeans.save_labels(labels, output_filename)

def process_mls(n_files = 100, model = None, model_filename = None):
    if model is None: model = kmeans.load_kmeans_model(model_filename)
    fn = locations.get_mls_filenames(n_files)
    for f in progressbar(fn[:n_files]):
        f = Path(f)
        output_filename = locations.kmeans_labels / (f.stem + '.txt')
        if output_filename.exists(): continue
        labels = kmeans.audio_filename_to_labels(f, model)
        kmeans.save_labels(labels, output_filename)

def process_cv(n_files = 1000, model = None, model_filename = None):
    if model is None: model = kmeans.load_kmeans_model(model_filename)
    fn = locations.get_cv_filenames(n_files)
    for f in progressbar(fn[:n_files]):
        f = Path(f)
        output_filename = locations.kmeans_labels / (f.stem + '.txt')
        if output_filename.exists(): continue
        labels = kmeans.audio_filename_to_labels(f, model)
        kmeans.save_labels(labels, output_filename)


