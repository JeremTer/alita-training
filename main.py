from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

FOLDER_COLUMN_INDEX = 1
FILENAME_COLUMN_INDEX = 0
CLASS_COLUMN_INDEX = 2


class AlitaSoundDataset(Dataset):

    def __init__(self, annotations_file: str, audio_dir: str):
        self.annotations = pd.read_csv(annotations_file, sep=';')
        self.audio_dir = audio_dir

    def __len__(self):  # how to calculate the length
        return len(self.annotations)

    def __getitem__(self, row: int):  # a[1] -> a.__getitem__(1), how to get item from dataset
        audio_sample_path = self._get_audio_sample_path(row)
        label = self._get_audio_sample_label(row)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, row: int) -> str:
        fold = f"folder{self.annotations.iloc[row, FOLDER_COLUMN_INDEX]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[row, FILENAME_COLUMN_INDEX])
        return path

    def _get_audio_sample_label(self, row: int) -> str:
        return self.annotations.iloc[row, CLASS_COLUMN_INDEX]


if __name__ == '__main__':
    ANNOTATIONS_FILE = '/Users/jerem/Dev/audios.csv'
    AUDIO_DIR = '/Users/jerem/Dev/audios'

    asd = AlitaSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f"There are {len(asd)} samples in the dataset.")
    print(asd)

    signal, label = asd[0]

    print(signal)
    print(label)
