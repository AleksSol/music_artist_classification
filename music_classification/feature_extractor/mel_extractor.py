import numpy as np
import librosa


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
        :param n_fft: FFT window size
        :param hop_length: number of samples between successive frames
        """

        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def transform(self, y: np.array) -> np.array:
        """

        :param y: numpy array[shape = (n,)] - audio time series
        :return: numpy array[shape = (n_mels, t) - resulting mel spectrogram
        """

        mel_unscaled = librosa.feature.melspectrogram(y, sr=self.sr, n_mels=self.n_mels,
                                                      n_fft=self.n_fft, hop_length=self.hop_length)
        mel_scaled = librosa.amplitude_to_db(mel_unscaled, ref=1.0)

        return mel_scaled
