import tensorflow as tf

from tensorflow.keras.layers import Layer


class Mfcc(Layer):

    """
    Custom (non-trainable) layer which computes MFCC for input audio signal.
    """

    def __init__(self, **kwargs):

        """
        Initialize necessary parameter
        :param kwargs: To set trainable=False.
        """

        super().__init__(**kwargs)
        self.sampling_frequency = 16000
        self.hop_length = int(self.sampling_frequency / 100)
        self.n_fft = 40
        self.lower_edge_hertz, self.upper_edge_hertz, self.num_mel_bins = 80.0, 7600.0, 40
        self.stfts = tf.signal.stft(frame_length=self.n_fft, frame_step=self.hop_length)
        self.spectrograms = tf.abs(self.stfts)
        self.num_spectrogram_bins = self.stfts.shape[-1]
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins,
            self.num_spectrogram_bins,
            self.sampling_frequency,
            self.lower_edge_hertz,
            self.upper_edge_hertz,
        )
        self.mel_spectrograms = tf.tensordot(
            self.spectrograms, self.linear_to_mel_weight_matrix, 1
        )
        self.mel_spectrograms.set_shape(
            self.spectrograms.shape[:-1].concatenate(self.linear_to_mel_weight_matrix.shape[-1:])
        )
        self.log_mel_spectrograms = tf.math.log(self.mel_spectrograms + 1e-6)
        self.features_tf = tf.signal.mfccs_from_log_mel_spectrograms(self.log_mel_spectrograms)

    def call(self, input_sig, fft_len=512):

        """
        MFCC computation
        :param input_sig: Input audio tensor
        :param fft_len: Size of FFT to apply
        :return: Computed MFCC tensor
        """

        x = self.stfts(input_sig, fft_len=fft_len)
        x = self.spectrograms(x)
        x = self.num_spectrogram_bins(x)
        x = self.linear_to_mel_weight_matrix(x)
        x = self.mel_spectrograms(x)
        x = self.log_mel_spectrograms(x)
        x = self.features_tf(x)

        return tf.expand_dims(tf.transpose(x), axis=-1)
