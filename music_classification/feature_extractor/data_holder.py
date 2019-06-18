import os
import pickle
from typing import Mapping

import librosa
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .mel_extractor import MelExtractor


class DataHolder:
    """Class for processing data"""

    def __init__(self, random_state: int = 127, sr: int = 16000, **kwargs):
        """

        :param random_state: random state to splitting data
        :param kwargs: parameters of MelExtractor
        """

        self.random_state = random_state
        self.extractor = MelExtractor(sr=sr, **kwargs)
        self.sr = sr
        self.artists_codes = {}
        self.codes_artists = {}
        self.songs_artists = {}
        self.songs_albums = {}
        self.songs_num_in_album = {}
        self.songs_codes = {}
        self.codes_songs = {}
        self.dataset = {}

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
                    self.codes_artists[artist_num] = artist_name
                    artist_num += 1
                    self.dataset[artist_name] = {}

                    with os.scandir(entry.path) as artist_folder:
                        for inner_entry in artist_folder:
                            if inner_entry.is_dir():
                                album_name = inner_entry.name
                                self.dataset[artist_name][album_name] = []

                                with os.scandir(inner_entry.path) as album_folder:
                                    num_in_album = 0
                                    for song in album_folder:
                                        if not song.name.startswith('.') and song.is_file() and song.name.endswith(
                                                ".mp3"):
                                            self.songs_artists[song.name] = artist_name
                                            self.songs_albums[song.name] = album_name
                                            song_code = len(self.songs_codes) + 1
                                            self.songs_num_in_album[song.name] = num_in_album
                                            num_in_album += 1
                                            self.songs_codes[song.name] = song_code
                                            self.codes_songs[song_code] = song.name

                                            song_raw, _ = librosa.load(song.path, sr=self.sr)
                                            song_mel = self.extractor.transform(song_raw)
                                            self.dataset[artist_name][album_name].append((song_mel,
                                                                                          self.artists_codes[
                                                                                              artist_name], song.name))

    def save_dataset(self, dump_path: str = 'dataset') -> None:
        """
        Save dataset to file

        :param dump_path: path to file
        :return:
        """

        with open(dump_path, "wb") as f_out:
            pickle.dump((self.dataset, self.artists_codes, self.codes_artists, self.songs_artists, self.songs_albums,
                         self.songs_num_in_album, self.songs_codes, self.codes_songs), f_out)

        return

    def load_dataset(self, dump_path: str = 'dataset') -> None:
        """
        Load dataset from file

        :param dump_path: path to file
        :return:
        """
        with open(dump_path, "rb") as f_in:
            (self.dataset, self.artists_codes, self.codes_artists, self.songs_artists, self.songs_albums,
             self.songs_num_in_album, self.songs_codes, self.codes_songs) = pickle.load(f_in)

        return

    @staticmethod
    def _slice_spectrogram(mel: np.array, slice_length: int = 900, overlap: int = 100) -> np.array:
        """
        Slice one spectrogram to parts

        :param mel: mel spectrogram of song np.array[shape=(mel_len, t)]
        :param slice_length: length of one slice
        :param overlap: overlap length
        :return: mel spectrograms of song parts np.array[shape=(parts, mel_len, slice_length)]
        """

        sliced_spectrogram = []
        length = mel.shape[1]

        slice_num = 0
        while slice_num * (slice_length - overlap) + slice_length <= length:
            cur_start = slice_num * (slice_length - overlap)
            sliced_spectrogram.append(mel[:, cur_start: cur_start + slice_length])
            slice_num += 1

        return np.array(sliced_spectrogram)

    def _build_sliced_set(self, songs=None, albums=None, slice_length: int = 900, overlap: int = 100):
        """
        Build a set of sliced spectrograms

        :param slice_length: length of one slice
        :param overlap: overlap: overlap length

        :return: a tuple (X, Y, names):
        X - np.array(shape=[samples, mel_len, slice_length)
        Y - np.array(shape=[samples,]) - int
        names - names of songs
        """

        sliced_set = []
        set_artists = []
        set_names = []

        if songs is not None and albums is not None:
            raise RuntimeError("Songs and albums arguments are not None simultaneously!")

        if songs is not None:
            for song in songs:
                artist = self.songs_artists[song]
                album = self.songs_albums[song]
                data = self.dataset[artist][album][self.songs_num_in_album[song]]
                sliced_mel = self._slice_spectrogram(data[0], slice_length=slice_length, overlap=overlap)

                num_of_parts = sliced_mel.shape[0]

                if num_of_parts != 0:  # Throw out songs shorter than one window
                    sliced_set.append(sliced_mel)
                    set_artists += [data[1]] * num_of_parts
                    set_names += [data[2]] * num_of_parts

        elif albums is not None:
            for artist in albums:
                for album in albums[artist]:
                    for song_data in self.dataset[artist][album]:
                        sliced_mel = self._slice_spectrogram(song_data[0], slice_length=slice_length, overlap=overlap)

                        num_of_parts = sliced_mel.shape[0]

                        if num_of_parts != 0:  # Throw out songs shorter than one window
                            sliced_set.append(sliced_mel)
                            set_artists += [song_data[1]] * num_of_parts
                            set_names += [song_data[2]] * num_of_parts
        else:
            raise RuntimeError("Songs and albums arguments are None simultaneously!")

        sliced_set = np.concatenate(sliced_set, axis=0)

        return sliced_set, np.array(set_artists), np.array(set_names)

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
        Y - np.array(shape=[samples,]) - int
        names - names of songs
        """

        stratified_split_test = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=self.random_state)
        stratified_split_val = StratifiedShuffleSplit(n_splits=2, test_size=val_size, random_state=self.random_state)

        songs_artists_items = self.songs_artists.items()
        songs_arr = np.array([song for song, artist in songs_artists_items])
        artists_arr = np.array([artist for song, artist in songs_artists_items])

        full_train_indices = []
        train_indices = []
        val_indices = []
        test_indices = []

        for full_train_indices, test_indices in stratified_split_test.split(songs_arr, artists_arr):
            for train_indices, val_indices in \
                    stratified_split_val.split(songs_arr[full_train_indices], artists_arr[full_train_indices]):
                break
            break

        train_songs = songs_arr[full_train_indices][train_indices]
        val_songs = songs_arr[full_train_indices][val_indices]
        test_songs = songs_arr[test_indices]

        train_set = self._build_sliced_set(songs=train_songs, albums=None, slice_length=slice_length, overlap=overlap)
        val_set = self._build_sliced_set(songs=val_songs, albums=None, slice_length=slice_length, overlap=overlap)
        test_set = self._build_sliced_set(songs=test_songs, albums=None, slice_length=slice_length, overlap=overlap)

        return {"train": train_set, "validation": val_set, "test": test_set}

    def get_album_split(self,
                        test_albums_num: int = 1,
                        val_albums_num: int = 1,
                        slice_length: int = 900,
                        overlap: int = 100) -> Mapping[str, tuple]:
        """
        Return album based split of data

        :param test_albums_num: number of albums in test for each artist
        :param val_albums_num: number of albums in validation for each artist
        :param slice_length: length of one slice
        :param overlap: overlap length

        :return: dict with keys 'train', 'validation', 'test' and values (X, Y, names)
        X - np.array(shape=[samples, mel_len, slice_length)
        Y - np.array(shape=[samples,]) - int
        names - names of songs
        """

        train_albums_dict = {}
        val_albums_dict = {}
        test_albums_dict = {}

        for artist in self.dataset.keys():
            albums = np.array(list(self.dataset[artist].keys()))
            num_of_albums = albums.shape[0]
            special_numbers = np.random.RandomState(seed=self.random_state).permutation(num_of_albums)
            train_numbers = special_numbers[val_albums_num + test_albums_num:]
            val_numbers = special_numbers[:val_albums_num]
            test_numbers = special_numbers[val_albums_num: val_albums_num + test_albums_num]

            train_albums_dict[artist] = albums[train_numbers]
            val_albums_dict[artist] = albums[val_numbers]
            test_albums_dict[artist] = albums[test_numbers]

        train_set = self._build_sliced_set(albums=train_albums_dict, slice_length=slice_length, overlap=overlap)
        val_set = self._build_sliced_set(albums=val_albums_dict, slice_length=slice_length, overlap=overlap)
        test_set = self._build_sliced_set(albums=test_albums_dict, slice_length=slice_length, overlap=overlap)

        return {"train": train_set, "validation": val_set, "test": test_set}

    @property
    def song_to_album(self) -> Mapping[str, str]:
        """

        :return: mapping from song name to album name
        """

        return self.songs_albums
