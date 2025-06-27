from pathlib import Path
import random

hubert_root=Path('/projects/0/prjs0893/speech-training/scripts/pretraining/hubert/')
kmeans_models = hubert_root / 'kmeans_models'
kmeans_model_filename = kmeans_models / 'km_train_mfcc-model_100clusters'

cluster_check = hubert_root / 'cluster_check'
kmeans_labels = cluster_check / 'kmeans_labels'

speech_training = Path('/projects/0/prjs0893/speech-training')

pretrain_datasets = speech_training / 'data/pretraining-datasets/'
data = speech_training / 'data'

cgn_news_sentences = data / 'cgn_sentences/split_files'

common_voice = pretrain_datasets / 'cv_wav'

mls = pretrain_datasets / 'MLS_dutch_wav'

pretrain_manifest = data / 'manifests/dataset-960h-nl/train.tsv'


metadata = Path('/projects/0/prjs0893/speech-training/metadata')
cgn_news_phonemes = metadata / 'news_books_phonemes_zs.tsv'
mls_phonemes = metadata / 'dutch_mls_phonemes_zs.tsv'
cv_phonemes = metadata / 'dutch_cv_phonemes_zs.tsv'


def load_manifest():
    with open(pretrain_manifest) as fin:
        manifest = fin.read().split('\n')
    return manifest[1:-1]

def get_mls_filenames(n_files = 100):
    random.seed(9)
    manifest = load_manifest()
    random.shuffle(manifest)
    output = [] 
    directory = pretrain_datasets
    for line in manifest:
        if not 'MLS' in line: continue
        f = directory / line.split('\t')[0]
        if not f.exists():
            raise ValueError(f'could not find {f}')
        output.append(str(f))
        if len(output) >= n_files:break
    return output

    
def get_cv_filenames(n_files = 1000):
    random.seed(9)
    manifest = load_manifest()
    random.shuffle(manifest)
    output = [] 
    directory = pretrain_datasets
    for line in manifest:
        if not 'common_voice' in line: continue
        f = directory / line.split('\t')[0]
        if not f.exists():
            raise ValueError(f'could not find {f}')
        output.append(str(f))
        if len(output) >= n_files:break
    return output


def corpus_to_kmeans_label_filenames():
    label_filenames = kmeans_labels.glob('*.txt')
    o = {'cgn':[], 'mls':[], 'cv':[]}
    for f in label_filenames:
        name = f.name
        if 'fn' in name or 'fv' in name: o['cgn'].append(f)
        elif 'common_voice' in name: o['cv'].append(f)
        else: o['mls'].append(f)
    return o
