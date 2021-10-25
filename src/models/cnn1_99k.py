import tensorflow.keras as tf

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    BatchNormalization,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from src.models.nn_models import NnModel


class Cnn1Param99k(NnModel):
    def __init__(
        self,
        N1,
        N2,
        kernel_size1,
        strides1,
        pool_size1,
        pool_stride1,
        kernel_size2,
        strides2,
        pool_size2,
        pool_stride2,
        Nfc1,
        Nfc2,
        Nfc3,
        dropout1,
        dropout2,
        dropout3,
    ):

        """

        :param N1:
        :param N2:
        :param kernel_size1:
        :param strides1:
        :param pool_size1:
        :param pool_stride1:
        :param kernel_size2:
        :param strides2:
        :param pool_size2:
        :param pool_stride2:
        :param Nfc1:
        :param Nfc2:
        :param Nfc3:
        :param dropout1:
        :param dropout2:
        :param dropout3:
        """

        self.N1 = N1
        self.N2 = N2
        self.kernel_size1 = kernel_size1
        self.strides1 = strides1
        self.pool_size1 = pool_size1
        self.pool_stride1 = pool_stride1
        self.kernel_size2 = kernel_size2
        self.strides2 = strides2
        self.pool_size2 = pool_size2
        self.pool_stride2 = pool_stride2
        self.Nfc1 = Nfc1
        self.Nfc2 = Nfc2
        self.Nfc3 = Nfc3
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        super().__init__()
        self.model_out = None

    def model_architecture(self):

        """

        :return:
        """

        model_input = Input(shape=self.input_shape)

        model = Conv2D(
            self.N1, kernel_size=self.kernel_size1, strides=self.strides1, activation="relu"
        )(model_input)

        model = BatchNormalization(axis=-1, scale=None)(model)

        model = MaxPooling2D(pool_size=self.pool_size1, strides=self.pool_stride1)(model)

        model = Conv2D(
            self.N2, kernel_size=self.kernel_size2, strides=self.strides2, activation="relu"
        )(model)

        model = BatchNormalization(axis=-1, scale=None)(model)

        model = MaxPooling2D(pool_size=self.pool_size2, strides=self.pool_stride2)(model)

        model = Flatten()(model)

        model = Dense(self.Nfc1, activation="relu")(model)

        model = Dropout(self.dropout1)(model)

        model = Dense(self.Nfc2, activation="relu")(model)

        model = Dropout(self.dropout2)(model)

        model = Dense(self.Nfc3, activation="relu")(model)

        model = Dropout(self.dropout3)(model)

        out = Dense(self.out, activation="softmax")(model)

        self.model_out = Model(inputs=[model_input], outputs=out)

    def model_compile(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> tf.Model:

        """
        Function to compile the model

        :param learning_rate: Initial learning rate for ADAM optimizer
        :param beta1: Exponential decay rate for the running average of the gradient
        :param beta2: Exponential decay rate for the running average of the square of the gradient
        :param epsilon: Epsilon parameter to prevent division by zero error
        :return: Compiled Keras model
        """

        adam = Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
        self.model_out.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=adam,
            metrics=["sparse_categorical_accuracy"],
        )

        return self.model_out
