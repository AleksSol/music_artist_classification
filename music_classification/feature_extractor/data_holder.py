from typing import Mapping
import numpy as np
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
        # TODO inner structure of DataHolder

    def process_folder(self, folder: str) -> None:
        """
        Process folder and save data to inner state

        :param folder: path to folder with format folder/artist/album/*.mp3
        :return:
        """
        pass

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
        Y - np.array(shape=[samples,]) - str
        names - names of songs
        """

    @property
    def song_to_album(self) -> Mapping[str, str]:
        """

        :return: mapping from song name to album name
        """
        return {}