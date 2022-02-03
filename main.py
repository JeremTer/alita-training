import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchsummary import summary

from cnn import CNNetwork

FOLDER_COLUMN_INDEX = 1
FILENAME_COLUMN_INDEX = 0
CLASS_COLUMN_INDEX = 2


class AlitaSoundDataset(Dataset):

    def __init__(self, annotations_file: str, audio_dir: str,
                 target_sample_rate: int, num_samples: int, transformation):
        self.annotations = pd.read_csv(annotations_file, sep=';')
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.transformation = transformation


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sample_rate)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        signal_length = signal.shape[1]
        if signal_length < self.num_samples:
            num_missing_samples = self.num_samples - signal_length
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _cut_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _resample_if_necessary(self, signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, row: int) -> str:
        fold = f"folder{self.annotations.iloc[row, FOLDER_COLUMN_INDEX]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[row, FILENAME_COLUMN_INDEX])
        return path

    def _get_audio_sample_label(self, row: int) -> str:
        return self.annotations.iloc[row, CLASS_COLUMN_INDEX]


if __name__ == '__main__':
    ANNOTATIONS_FILE = 'datasets/audios.csv'
    AUDIO_DIR = 'datasets'
    SAMPLE_RATE = 44100
    NUM_SAMPLES = 90000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    asd = AlitaSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, mel_spectrogram)
    signal, label = asd[1]
    print(signal.shape)

    cnn = CNNetwork()
    summary(cnn, (1, 64, 176))
