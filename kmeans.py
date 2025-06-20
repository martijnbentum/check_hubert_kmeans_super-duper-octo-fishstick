import audio
import joblib
import locations

def load_kmeans_model(filename = None):
    if filename is None:
        filename = locations.kmeans_model_filename
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
        
