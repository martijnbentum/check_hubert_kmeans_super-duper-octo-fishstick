import copy
import numpy as np

class Frames:
    '''Frames class to handle the frames of kmeans labels
    '''
    def __init__(self,n_frames, stride = 0.01, field = 0.025, 
        start_time = 0, frames = None, identifier = '', name = '',
        audio_filename = '', labels = None):
        '''Handles the frames 
        n_frames        number of frames
        stride          the time between frames
        field           the length of the frame
        start_time      the start time of the first frame
        frames          list of frames
        identifier      identifier of the labels
        name            name of the labels
        audio_filename  audio filename
        labels          the labels of a model
        '''
        
        self.identifier = identifier
        self.name = name
        self.audio_filename = audio_filename
        self.labels = labels 
        self.stride = stride
        self.field = field
        self.n_frames = n_frames
        self.start_time = start_time
        self._make_frames()
        self.end_time = self.frames[-1].end_time 
        self.duration = self.end_time - self.frames[0].start_time

    def __repr__(self):
        m = f'Frames(#frames:{self.n_frames}, start:{self.start_time:.3f}'
        m += f', duration:{self.duration:.3f}'
        m += f', stride:{self.stride:.3f}, field:{self.field})'
        return m

    def _make_frames(self):
        self.frames = []
        for index in range(self.n_frames):
            self.frames.append(Frame(index, self.stride, self.field, 
                self.start_time, self))

    def start_frame(self, start = None, end = None, percentage_overlap = None):
        if start == end == percentage_overlap == None:
            return select_start_frame(self.frames)
        if start is None:
            start = self.start_time
        if end is None:
            end = self.end_time
        selected_frames = self.select_frames(start, end, 
            percentage_overlap = percentage_overlap)
        return select_start_frame(selected_frames)
        
    def middle_frame(self, start = None, end = None, percentage_overlap = None):
        '''Get the middle frame of the frames that overlap with the start 
        and end times
        '''
        if start == end == percentage_overlap == None:
            return select_middle_frame(self.frames)
        if start is None:
            start = self.start_time
        if end is None:
            end = self.end_time
        selected_frames = self.select_frames(start, end, 
            percentage_overlap = percentage_overlap)
        return select_middle_frame(selected_frames)

    def end_frame(self, start = None, end = None, percentage_overlap = None):
        '''Get the end frame of the frames that overlap with the start 
        and end times
        '''
        if start == end == percentage_overlap == None:
            return select_end_frame(self.frames)
        if start is None:
            start = self.start_time
        if end is None:
            end = self.end_time
        selected_frames = self.select_frames(start, end, 
            percentage_overlap = percentage_overlap)
        return select_end_frame(selected_frames)

    def start_middle_end_frames(self, start = None, end = None,
        percentage_overlap = None):
        '''Get the start, middle and end frames of the frames that overlap 
        with the start and end times
        '''
        if start == end == percentage_overlap == None:
            return select_start_middle_end_frames(self.frames)
        if start is None:
            start = self.start_time
        if end is None:
            end = self.end_time
        selected_frames = self.select_frames(start, end, 
            percentage_overlap = percentage_overlap)
        return select_start_middle_end_frames(selected_frames)

    def select_frames(self,start = None,end = None, percentage_overlap = None):
        '''Select frames that overlap with the start and end times
        start               start time
        end                 end time
        percentage_overlap  the percentage of the frame that must overlap
                            with the segment between start and end time
        '''
        if start == end == None:
            return self.frames
        if start is None: start = self.start_time
        if end is None: end = self.end_time
        selected_frames = []
        for frame in self.frames:
            if percentage_overlap == None:
                if frame.overlap(start,end):
                    selected_frames.append(frame)
            elif frame.overlap_percentage(start,end) >= percentage_overlap:
                selected_frames.append(frame)
        return selected_frames


    def label(self, start_time = None, end_time = None, 
        percentage_overlap = None, middle_frame = False):
        '''Get the transformer output of the frames that overlap with the start 
        and end times
        '''
        if self.label is None: return None
        if middle_frame:
            frame = self.middle_frame(start_time, end_time,
                percentage_overlap = percentage_overlap)
            return frame.label
        frames = self.select_frames(start_time, end_time, 
            percentage_overlap = percentage_overlap)
        if len(frames) == 1: return frames[0].label
        labels = [frame.label() for frame in frames]
        return labels


class Frame:
    '''Frame class to handle a single frame of the kmeans labels.'''
    def __init__(self, index, stride, field, global_start_time, parent):
        '''Handles a single frame of the kmeans labels
        index               index of the frame
        stride              the time between frames
        field               the length of the frame
        global_start_time   the start time of the first frame
        parent              the parent Frames object
        '''
        self.index = index
        self.stride = stride
        self.field = field
        self.global_start_time = global_start_time
        self.parent = parent
            
        self.start_time = self.index * self.stride + self.global_start_time
        self.end_time = self.start_time + self.field

    def __repr__(self):
        return f'Frame({self.index}, {self.start_time:.3f}, {self.end_time:.3f})'

    def overlap(self,start,end):
        return self.start_time < end and self.end_time > start

    def overlap_time(self,start,end): 
        return min(self.end_time,end) - max(self.start_time,start)

    def overlap_percentage(self,start,end):
        if self.overlap(start,end):
            ot = self.overlap_time(start,end)
            return round(ot / self.field,5) * 100
        return 0.0

    def label(self):
        if not hasattr(self.parent,'labels'):
            return None
        return self.parent.labels[self.index]



def find_frame_start_time(start_time):
    '''find the start time of the first frame.'''
    if abs(0 - start_time) < 0.001: return 0
    stride = 0.02
    end_time = start_time + 0.001
    nframes = int(start_time / stride) + 5
    frames = Frames(nframes, start_time =  0)
    for i, f in enumerate(frames.frames):
        if f.overlap(start_time, end_time): 
            return f.start_time
    raise ValueError('Could not find frame start time', start_time)

    
def select_start_frame(frames):
    n_frames = len(frames)
    if n_frames > 0: return frames[0]

def select_middle_frame(frames):
    n_frames = len(frames)
    if n_frames == 1: return frames[0]
    if n_frames % 2 == 0:
        return frames[int(n_frames / 2) - 1]
    return frames[int(n_frames / 2)]

def select_end_frame(frames):
    n_frames = len(frames)
    if n_frames > 0: return frames[-1]

def select_start_middle_end_frames(frames):
    n_frames = len(frames)
    d = {'start': None, 'middle': None, 'end': None}
    if n_frames == 0: return d
    if n_frames > 0:
        d['start'] = select_start_frame(frames)
    if n_frames > 1:
        d['end'] = select_end_frame(frames)
    if n_frames == 3:
        d['middle'] = frames[1]
    elif n_frames > 3:
        d['middle'] = select_middle_frame(frames)
    return d

        
