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

        super(Mfcc, self).__init__(**kwargs)
        self.sampling_frequency = 16000
        self.hop_length = int(self.sampling_frequency / 100)
        self.n_fft = 40
        self.fft_len = 512
        self.lower_edge_hertz, self.upper_edge_hertz, self.num_mel_bins = 80.0, 7600.0, 40

    def call(self, input_sig, fft_len=512):

        """
        MFCC computation

        :param input_sig: Input audio tensor
        :param fft_len: Size of FFT to apply
        :return: Computed MFCC tensor
        """

        stfts = tf.signal.stft(
            input_sig, frame_length=self.n_fft, frame_step=self.hop_length, fft_length=fft_len
        )
        spectrograms = tf.abs(stfts)

        num_spectrogram_bins = stfts.shape[-1]

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins,
            num_spectrogram_bins,
            self.sampling_frequency,
            self.lower_edge_hertz,
            self.upper_edge_hertz,
        )
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        features_tf = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)

        return tf.expand_dims(tf.transpose(features_tf), axis=-1)
