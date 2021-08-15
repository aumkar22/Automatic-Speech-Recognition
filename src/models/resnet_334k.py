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
from src.models.mfcc_layer import Mfcc
from src.models.residual_layer import Residual


class ResNetParam334k(NnModel):
    def __init__(
        self,
        N1,
        N2,
        kernel_size1,
        strides1,
        pool_size1,
        kernel_size2,
        dilation_rate1,
        dilation_rate2,
        dilation_rate3,
        dilation_rate4,
        Nfc1,
        Nfc2,
        dropout1,
        dropout2,
    ):

        """

        :param N1:
        :param N2:
        :param kernel_size1:
        :param strides1:
        :param pool_size1:
        :param kernel_size2:
        :param dilation_rate1:
        :param dilation_rate2:
        :param dilation_rate3:
        :param dilation_rate4:
        :param Nfc1:
        :param Nfc2:
        :param dropout1:
        :param dropout2:
        """

        self.N1 = N1
        self.N2 = N2
        self.kernel_size1 = kernel_size1
        self.strides1 = strides1
        self.pool_size1 = pool_size1
        self.kernel_size2 = kernel_size2
        self.Nfc1 = Nfc1
        self.Nfc2 = Nfc2
        self.dilation_rate1 = dilation_rate1
        self.dilation_rate2 = dilation_rate2
        self.dilation_rate3 = dilation_rate3
        self.dilation_rate4 = dilation_rate4
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        super().__init__()
        self.model_out = None

    def model_architecture(self):

        """

        :return:
        """

        model_input = Input(shape=self.input_shape)

        model = Conv2D(
            self.N1,
            kernel_size=self.kernel_size1,
            strides=self.strides1,
            activation="relu",
            padding="same",
        )(model_input)

        model = BatchNormalization(axis=-1, scale=None)(model)

        model = Residual(self.N1, self.kernel_size1, self.dilation_rate1)(model)

        model = Residual(self.N1, self.kernel_size1, self.dilation_rate2)(model)

        model = Residual(self.N1, self.kernel_size1, self.dilation_rate3)(model)

        model = Conv2D(
            self.N2,
            kernel_size=self.kernel_size2,
            activation="relu",
            dilation_rate=self.dilation_rate4,
        )(model)

        model = BatchNormalization(axis=-1, scale=None)(model)

        model = MaxPooling2D(pool_size=self.pool_size1)(model)

        model = Flatten()(model)

        model = Dense(self.Nfc1, activation="relu")(model)

        model = Dropout(self.dropout1)(model)

        model = Dense(self.Nfc2, activation="relu")(model)

        model = Dropout(self.dropout2)(model)

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

        adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
        self.model_out.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=adam,
            metrics=["sparse_categorical_accuracy"],
        )

        return self.model_out
