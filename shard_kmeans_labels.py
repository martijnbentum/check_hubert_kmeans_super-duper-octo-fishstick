import locations
import manifest
from pathlib import Path

class AllLabels:
    def __init__(self, filenames = None, manifest_data = None, 
        feature_sample_rate = None):
        self._set_filenames(filenames)
        if manifest_data is None:
            _, self.manifest_data = manifest.load_manifest()
        else: self.manifest_data = manifest_data
        self.feature_sample_rate = feature_sample_rate


    def _set_filenames(self, filenames):
        temp = []
        self.train_filename = ''
        for f in filenames:
            f = Path(f)
            if f.stem == 'train': self.train_filename = f
            else: temp.append(f)
        self.filenames= sorted(temp, key = lambda x: int(str(x).split('_')[-2]))

    def _load(self):
        self.recording_labels = []
        self.manifest_index = 0
        for filename in self.filenames:
            self._handle_shard_filename(filename)

    def _handle_shard_filename(self,filename):
        print(f'loading {filename}')
        o = load_shard_kmean_labels(filename)
        self.big_diff = []
        for line in o:
            manifest_line = self.manifest_data[self.manifest_index]
            rl = RecordingLabels(line, manifest_line, self.feature_sample_rate)
            if rl.diff >= 2:
                print(rl)
                self.big_diff.append(rl)
            self.recording_labels.append(rl)
            self.manifest_index += 1

class RecordingLabels:
    def __init__(self, labels, manifest_line= None, feature_sample_rate = None):
        self.labels = labels
        self.n_labels = len(labels)
        self.feature_sample_rate = feature_sample_rate
        self.set_manifest_line(manifest_line)

    def __repr__(self):
        m = f'nlabels: {self.n_labels}'
        m += f' n_frames: {self.n_frames}'
        m += f' diff: {self.diff}'
        if self.audio_filename:
            m += f' filename: {self.audio_filename.name}'
        return m
    

    def set_manifest_line(self, manifest_line):
        self.audio_filename = None
        self.n_samples = None
        self.n_frames = None
        self.diff = None
        if manifest_line is not None: 
            self.audio_filename = Path(manifest_line[0])
            self.n_samples = manifest_line[1] 
            if self.feature_sample_rate is not None:
                nf, dur = manifest.manifest_line_to_number_of_frames(
                    manifest_line,
                    feature_sample_rate = self.feature_sample_rate)
                self.n_frames = nf
                self.duration = dur
                self.diff = abs(self.n_frames - self.n_labels)
            
            



def load_shard_kmean_labels(filename):
    with open(filename) as fin:
        temp= fin.read().split('\n')
    recording_labels = []
    for line in temp:
        labels = [int(label) for label in line.split(' ') if label]
        if labels == []: continue
        recording_labels.append(labels)
    return recording_labels

