import numpy as np


class MelExtractor:
    """
    Class to extract mel-spectrogram features from audio
    """

    def __init__(self,
                 sr: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512
                 ):
        """
        Init parameters of mel transform

        :param sr: sample rate
        :param n_mels: num of mels
        :param n_fft: length of the FFT window
        :param hop_length: number of samples between successive frames
        """
        pass

    def transform(self, y: np.array) -> np.array:
        """

        :param y: numpy array[shape = (n,)] - audio time series
        :return: numpy array[shape = (n_mels, t) - resulting mel spectrogram
        """
        pass
