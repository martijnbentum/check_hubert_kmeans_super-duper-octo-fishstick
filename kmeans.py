import audio
import frames
import joblib
import locations
from pathlib import Path

def load_kmeans_model(filename = None, nclusters = 100, features = 'mfcc'):
    if nclusters not in [100, 500]:
        raise ValueError(nclusters, 'should be 100 or  500')
    if features not in ['mfcc','hub']:
        raise ValueError(features, 'should be mfcc or hub')
    if filename is None:
        attr_name = f'kmeans_model_filename_{features}{nclusters}'
        filename = getattr(locations,attr_name)
    print(f'loading model {filename}')
    model = joblib.load(str(filename))
    return model

def audio_to_labels(wav, model = None, model_filename = None):
    if model is None: model = load_kmeans_model(model_filename)
    mfccs = audio.wav_to_mfccs(wav)
    labels = model.predict(mfccs)
    return labels

def audio_filename_to_labels(audio_filename, model = None, 
    model_filename = None):
    wav = audio.load_audio(audio_filename)
    labels = audio_to_labels(wav, model, model_filename)
    return labels

def save_labels(labels, filename):
    output = ' '.join(map(str, labels))
    with open(filename, 'w') as fout:
        fout.write(output)

def load_labels(filename):
    with open(filename) as fin:
        labels = fin.read().split(' ')
    return list(map(int, labels))
        
def label_filename_to_frame_labels(filename, audio_filename = ''):
    labels = load_labels(filename)
    identifier = Path(filename).stem
    f = frames.Frames(len(labels), name = filename, identifier = identifier,
        audio_filename = audio_filename, labels = labels)
    return f
