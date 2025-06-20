import librosa
import torch
import torchaudio
import numpy

def load_audio(filename, sample_rate = 16000):
    wav, sr = librosa.load(filename, sr = sample_rate)
    return wav

def wav_to_mfccs(wav, sample_rate = 16000):
    with torch.no_grad():
        x = torch.from_numpy(wav).float()
        xx = x.view(1,-1)
        mfccs = torchaudio.compliance.kaldi.mfcc(waveform = xx, 
            sample_frequency=16000, use_energy=False)
        mfccs_ = mfccs.transpose(0,1)
        deltas = torchaudio.functional.compute_deltas(mfccs_)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs_, deltas, ddeltas], dim = 0)
        concat_ = concat.transpose(0,1).contiguous()
        output = concat_.numpy()
    return output

def filename_to_mfccs(filename, sample_rate = 16000):
    wav = load_audio(filename, sample_rate)
    mfccs = wav_to_mfccs(wav, sample_rate)
    return mfccs
