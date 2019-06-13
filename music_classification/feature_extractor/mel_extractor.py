import numpy as np
import librosa


class MelExtractor:
    """
    Class to extract mel-spectrogram features from audio
    """

    def __init__(self,
                 sr: int = 16000, # 16 * 1024?
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: int = 2048,
                 min_frequency: int = 50,
                 max_frequency: int = 32000
                 ):
        """
        Init parameters of mel transform

        :param sr: sample rate
        :param n_mels: num of mels
        :param n_fft: FFT window size
        :param hop_length: number of samples between successive frames
        :param win_length: length of the FFT window
        :param min_frequency: minimal mel frequency
        :param max_frequency: maximal mel frequency
        """

        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def transform(self, y: np.array) -> np.array:
        """

        :param y: numpy array[shape = (n,)] - audio time series
        :return: numpy array[shape = (n_mels, t) - resulting mel spectrogram
        """

        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

        mel_basis = librosa.filters.mel(self.sr, self.n_fft, n_mels=self.n_mels,
                                        fmin=self.min_frequency, fmax=self.max_frequency)

        mel_unscaled = np.dot(mel_basis, stft)

        # mel_unscaled = 2595 * np.log10(1 + stft / 700) # HARDCODE
        mel_scaled = 10 * np.log10(mel_unscaled / 1) # HARDCODE

        return mel_scaled
