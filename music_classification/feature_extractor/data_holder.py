from typing import Mapping
import numpy as np
import librosa
import os
import pickle
from .mel_extractor import MelExtractor
from sklearn.model_selection import StratifiedShuffleSplit


class DataHolder:
    """Class for processing data"""

    def __init__(self, random_state: int = 127, **kwargs):
        """

        :param random_state: random state to splitting data
        :param kwargs: parameters of MelExtractor
        """
        self.random_state = random_state
        self.extractor = MelExtractor(**kwargs)
        self.sr = kwargs["sr"]
        self.artists_codes = {}
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
                                        if not song.name.startswith('.') and song.is_file():
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
                                                                                          self.artists_codes[artist_name],
                                                                                          song.name))

                                            del song_raw
                                            del song_mel


    def save_dataset(self, dump_path: str = 'dataset') -> None:
        """
        Save dataset to file

        :param dump_path: path to file
        :return:
        """
        # TODO chose library to dump data (pickle, dill, json, etc)
        with open(dump_path, "wb") as f_out:
            pickle.dump((self.dataset, self.artists_codes, self.songs_artists, self.songs_albums,
                         self.songs_num_in_album, self.songs_codes, self.codes_songs), f_out)

        return

    def load_dataset(self, dump_path: str = 'dataset') -> None:
        """
        Load dataset from file

        :param dump_path: path to file
        :return:
        """
        with open(dump_path, "rb") as f_in:
            (self.dataset, self.artists_codes, self.songs_artists, self.songs_albums, self.songs_num_in_album,
             self.songs_codes, self.codes_songs) = pickle.load(f_in)

        return

    def _slice_spectrogram(self, mel: np.array, slice_length: int = 900, overlap: int = 100) -> np.array:
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

    def _build_sliced_set(self, songs, slice_length: int = 900, overlap: int = 100):
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

        for song in songs:
            artist = self.songs_artists[song]
            album = self.songs_albums[song]
            data = self.dataset[artist][album][self.songs_num_in_album[song]]
            sliced_mel = self._slice_spectrogram(data[0], slice_length=slice_length, overlap=overlap)

            num_of_parts = sliced_mel.shape[0]

            if num_of_parts != 0: # Throw out songs shorter than one window
                sliced_set.append(sliced_mel)
                set_artists += [data[1]] * num_of_parts
                set_names += [data[2]] * num_of_parts

        sliced_set = np.concatenate(sliced_set, axis=0)

        return (sliced_set, np.array(set_artists), np.array(set_names))

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
        # TODO create the same part of code with album split and move to new private method

        stratified_split_test = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=self.random_state)
        stratified_split_val = StratifiedShuffleSplit(n_splits=2, test_size=val_size, random_state=self.random_state)

        songs_artists_items = self.songs_artists.items()
        songs_arr = np.array([song for song, artist in songs_artists_items])
        artists_arr = np.array([artist for song, artist in songs_artists_items])

        for full_train_indices, test_indices in stratified_split_test.split(songs_arr, artists_arr):
            for train_indices, val_indices in\
            stratified_split_val.split(songs_arr[full_train_indices], artists_arr[full_train_indices]):
                break
            break

        train_songs = songs_arr[full_train_indices][train_indices]
        val_songs = songs_arr[full_train_indices][val_indices]
        test_songs = songs_arr[test_indices]

        train_set = self._build_sliced_set(train_songs, slice_length=slice_length, overlap=overlap)
        val_set = self._build_sliced_set(val_songs, slice_length=slice_length, overlap=overlap)
        test_set = self._build_sliced_set(test_songs, slice_length=slice_length, overlap=overlap)

        return {"train": train_set, "validation": val_set, "test": test_set}

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

        ## ??
        return {}

        # song_album_map = {}
        #
        # for artist in self.dataset.keys():
        #     for album in self.dataset[artist].keys():
        #         for song_info in self.dataset[artist][album]:
        #             song_album_map[song_info[2]] = album
        # return song_album_map