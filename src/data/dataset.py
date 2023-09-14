import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
import sys
from src.data.char_list import char_list
from src.data.transforms import *
from src.data.visual_corruption import *
import librosa
import cv2
import pickle

class AVDataset(Dataset):
    def __init__(self, data_path, split_file, mode, data_type, max_vid_len=600, max_txt_len=200, visual_corruption=True, fast_validate=False):
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.fast_validate = fast_validate
        self.visual_corruption = visual_corruption if mode == 'train' else False
        self.data_path = data_path
        self.file_paths = self.build_file_list(split_file, mode, data_type)
        self.char_list = char_list
        self.char2idx = {v: k for k, v in enumerate(char_list)}

        self.max_vid_len = max_vid_len
        self.max_txt_len = max_txt_len

        self._noise = np.load('./src/data/babbleNoise_resample_16K.npy')
        self.transform_aud = self.get_audio_transform(self._noise, split=mode)
        self.transform_vid = self.get_video_transform(split=mode)

        if visual_corruption:
            self.visual_corruption = Visual_Corruption_Modeling()

    def build_file_list(self, split_file, mode, data_type):
        datalist = open(split_file).read().splitlines()
        if mode == 'val':
            if data_type == 'LRS2':
                return [os.path.join('main', x.strip().split()[0]) for x in datalist]
            elif data_type == 'LRS3':
                return [os.path.join('trainval', x.strip().split()[0]) for x in datalist]
            else:
                raise NotImplementedError("data_type should be LRS2 or LRS3")
        else:
            return [x.strip().split()[0] for x in datalist]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        f_name = self.file_paths[idx]

        vid_path = os.path.join(self.data_path, 'Video', f_name + '.mp4')
        aud_path = os.path.join(self.data_path, 'Audio', f_name + '.wav')
        txt_path = os.path.join(self.data_path, 'Text', f_name + '.txt')
        lm_path = os.path.join(self.data_path, 'Transformed_LM', f_name + '.pkl')
        if not (os.path.exists(vid_path) and os.path.exists(aud_path) and os.path.exists(txt_path) and os.path.exists(lm_path)):
            return None
        
        vid = self.load_video(vid_path)
        if len(vid) == 0:
            return None
        if len(vid) > self.max_vid_len:
            vid = vid[:self.max_vid_len]
        
        aud = self.load_audio(aud_path)
        aud = aud[:int(len(vid) / 25 * 16000)]
        if len(aud) < int(len(vid) / 25 * 16000):
            aud = np.concatenate([aud, np.zeros(int(len(vid) / 25 * 16000) - len(aud)), 0])

        gt = self.load_txt(txt_path)
        text = np.array(self.parse_transcript(gt))
        if len(text) > self.max_txt_len:
            text = text[:self.max_txt_len]

        if self.visual_corruption:
            with open(lm_path, "rb") as pkl_file:
                pkl = pickle.load(pkl_file)
            lm = pkl['landmarks']
            yx_min = pkl['yx_min']

            prob = random.random()
            if prob < 0.2:
                pass
            else:
                freq1, freq2 = [random.choice([1, 2, 3]) for _ in range(2)]
                vid, _ = self.visual_corruption.occlude_sequence(vid, lm, yx_min, freq=freq1)
                vid = self.visual_corruption.noise_sequence(vid, freq=freq2)
        
        vid = self.transform_vid(vid)
        aud = self.transform_aud(aud)

        return vid, aud, text

    def load_txt(self, filename):
        text = open(filename, 'r').readline().strip()
        return text[text.find(' '):].strip()
    
    def parse_transcript(self, transcript):
        idx_list = list()
        for char in transcript:
            if char==" ":
                char = "<space>"
            idx = self.char2idx.get(char)
            idx = idx if idx is not None else self.char2idx['<unk>']
            idx_list.append(idx)
        return idx_list

    def get_audio_transform(self, noise_data, split):
        """get_audio_transform.

        :param noise_data: numpy.ndarray, the noisy data to be injected to data.
        """
        if split == 'test':
            return Compose([
                NormalizeUtterance(),
                ExpandDims()]
            )
        else:
            return Compose([
                AddNoise(
                    noise=noise_data
                ),
                NormalizeUtterance(),
                ExpandDims()]
            )

    def get_video_transform(self, split='test'):
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        return Compose([
            Normalize(0.0, 255.0),
            CenterCrop(crop_size) if split=='test' else RandomCrop(crop_size),
            Identity() if split=='test' else HorizontalFlip(0.5),
            Normalize(mean,std),
            Identity() if split=='test' else TimeMask(max_mask_length=15),
            Identity() if split=='test' else CutoutHole(min_hole_length=22, max_hole_length=44)
            ])

    def load_data(self, vid_path, aud_path):
        vid = self.load_video(vid_path)        
        aud = self.load_audio(aud_path)
        aud = aud[:int(len(vid) / 25 * 16000)]
        if len(aud) < int(len(vid) / 25 * 16000):
            aud = np.concatenate([aud, np.zeros(int(len(vid) / 25 * 16000) - len(aud)), 0])        
        vid = self.transform_vid(vid)
        aud = self.transform_aud(aud)
        return vid, aud

    def load_video(self, data_filename):
        frames = []
        cap = cv2.VideoCapture(data_filename)                           
        while(cap.isOpened()):                                                       
            ret, frame = cap.read() # BGR                                           
            if ret:                   
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                                
                frames.append(frame)
            else:
                break
        cap.release()
        return np.array(frames)

    def load_audio(self, audio_filename, specified_sr=16000, int_16=False):
        """load_audio.

        :param audio_filename: str, the filename for an audio waveform.
        :param specified_sr: int, expected sampling rate, the default value is 16KHz.
        :param int_16: boolean, return 16-bit PCM if set it as True.
        """
        try:
            if audio_filename.endswith('npy'):
                audio = np.load(audio_filename)
            elif audio_filename.endswith('npz'):
                audio = np.load(audio_filename)['data']
            else:
                audio, sr = librosa.load(audio_filename, sr=None)
                audio = librosa.resample(audio, sr, specified_sr) if sr != specified_sr else audio
        except IOError:
            sys.exit()

        if int_16 and audio.dtype == np.float32:
            audio = ((audio - 1.) * (65535. / 2.) + 32767.).astype(np.int16)
            audio = np.array(np.clip(np.round(audio), -2 ** 15, 2 ** 15 - 1), dtype=np.int16)
        if not int_16 and audio.dtype == np.int16:
            audio = ((audio - 32767.) * 2 / 65535. + 1).astype(np.float32)
        return audio

    def collate_fn(self, batch):
        vid_len, aud_len, text_len = [], [], []
        for data in batch:
            if data is not None:
                vid_len.append(len(data[0]))
                aud_len.append(len(data[1]))
                text_len.append(len(data[2]))

        max_vid_len = max(vid_len)
        max_aud_len = max(aud_len)
        max_text_len = max(text_len)

        padded_vid = []
        padded_aud = []
        padded_text = []

        for i, (vid, aud, text) in enumerate(batch):
            if data is not None:
                padded_vid.append(torch.cat([torch.tensor(vid), torch.zeros([max_vid_len - len(vid), 88, 88])], 0))
                padded_aud.append(torch.cat([torch.tensor(aud), torch.zeros([max_aud_len - len(aud), 1])], 0))
                padded_text.append(torch.cat([torch.tensor(text), torch.ones([max_text_len - len(text)]) * -1], 0))

        vid = torch.stack(padded_vid, 0).float()
        aud = torch.stack(padded_aud, 0).float()
        text = torch.stack(padded_text, 0).long()
        vid_len = torch.IntTensor(vid_len)
        aud_len = torch.IntTensor(aud_len)
        return vid, aud, vid_len, aud_len, text
