from typing import Mapping
import numpy as np
import librosa
import os
from .mel_extractor import MelExtractor


class DataHolder:
    """Class for processing data"""

    def __init__(self, random_state: int = 13, **kwargs):
        """

        :param random_state: random state to splitting data
        :param kwargs: parameters of MelExtractor
        """
        self.random_state = random_state
        self.extractor = MelExtractor(**kwargs)
        self.sr = kwargs["sr"]
        self.artists_codes = {}
        self.dataset = {}
        # TODO inner structure of DataHolder

    def process_folder(self, folder: str) -> None:
        """
        Process folder and save data to inner state

        :param folder: path to folder with format folder/artist/album/*.mp3
        :return:
        """

        with os.scandir(folder) as root:
            artist_num = 0
            for entry in root:
                if entry.is_dir():
                    artist_name = entry.name
                    self.artists_codes[artist_name] = artist_num
                    artist_num += 1
                    self.dataset[artist_name] = {}

                    with os.scandir(entry.path) as artist_folder:
                        for inner_entry in artist_folder:
                            if inner_entry.is_dir():
                                album_name = inner_entry.name
                                self.dataset[artist_name][album_name] = []

                                with os.scandir(inner_entry.path) as album_foder:
                                    for song in album_foder:
                                        if not song.name.startswith('.') and song.is_file():
                                            song_raw, _ = librosa.load(song.path, sr=self.sr)
                                            self.dataset[artist_name][album_name].append((song_raw,
                                                                                          self.artists_codes[artist_name],
                                                                                          song.name))

    def save_dataset(self, dump_path: str = 'dataset') -> None:
        """
        Save dataset to file

        :param dump_path: path to file
        :return:
        """
        # TODO chose library to dump data (pickle, dill, json, etc)

    def load_dataset(self, dump_path: str = 'dataset') -> None:
        """
        Load dataset from file

        :param dump_path: path to file
        :return:
        """

    def _slice_spectrogram(self, mel: np.array, slice_length: int = 900, overlap: int = 100) -> np.array:
        """
        Slice one spectrogram to parts

        :param mel: mel spectrogram of song np.array[shape=(mel_len, t)]
        :param slice_length: length of one slice
        :param overlap: overlap length
        :return: mel spectrograms of song parts np.array[shape=(parts, mel_len, slice_length)]
        """

    def get_song_split(self,
                       test_size: float = 0.1,
                       val_size: float = 0.1,
                       slice_length: int = 900,
                       overlap: int = 100) -> Mapping[str, tuple]:
        """
        Return song based split of data

        :param test_size: fraction of test data
        :param val_size: fraction of train data
        :param slice_length: length of one slice
        :param overlap: overlap length

        :return: dict with keys 'train', 'validation', 'test' and values (X, Y, names)
        X - np.array(shape=[samples, mel_len, slice_length)
        Y - np.array(shape=[samples,]) - str
        names - names of songs
        """
        # TODO create same part of code with album split and move to new private method

    def get_album_split(self,
                        test_size: float = 0.1,
                        val_size: float = 0.1,
                        slice_length: int = 900,
                        overlap: int = 100) -> Mapping[str, tuple]:
        """
        Return song based split of data

        :param test_size: fraction of test data
        :param val_size: fraction of train data
        :param slice_length: length of one slice
        :param overlap: overlap length

        :return: dict with keys 'train', 'validation', 'test' and values (X, Y, names)
        X - np.array(shape=[samples, mel_len, slice_length)
        Y - np.array(shape=[samples,]) - int
        names - names of songs
        """

    @property
    def song_to_album(self) -> Mapping[str, str]:
        """

        :return: mapping from song name to album name
        """
        return {}