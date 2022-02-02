import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

FOLDER_COLUMN_INDEX = 1
FILENAME_COLUMN_INDEX = 0
CLASS_COLUMN_INDEX = 2


class AlitaSoundDataset(Dataset):

    def __init__(self, annotations_file: str, audio_dir: str,
                 target_sample_rate: int):
        self.annotations = pd.read_csv(annotations_file, sep=';')
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int):
        audio_sample_path = self._get_audio_sample_path(index)
        print(audio_sample_path)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sample_rate)
        signal = self._mix_down_if_necessary(signal)
        return signal, label

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
    SAMPLE_RATE = 16000

    asd = AlitaSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE)
    sum = 0

    print(f"There are {len(asd)} samples in the dataset.")
    print(asd)

    for i in range(len(asd)):
        signal, label = asd[i]
        sum += signal.shape[1]
        print(signal.size())

print(sum / len(asd))
signal, label = asd[0]
print(signal)
print(label)
