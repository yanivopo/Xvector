import keras
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten, BatchNormalization
from keras import Model
#from keras.utils import plot_model


class Cnn:
    def __init__(self, data_dim=(512, 299), activation='relu', initializer='glorot_uniform', layer_size=[10, 16, 20, 100]):
        self.data_dim = data_dim
        self.layer_size = layer_size
        self.activation = activation
        self.initializer = initializer
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(*self.data_dim, 1))
        conv1 = Conv2D(filters=self.layer_size[0], kernel_size=(3, 3), activation=self.activation,
                       kernel_initializer=self.initializer, name='conv1')(inputs)
        conv1 = BatchNormalization()(conv1)
        maxpool1 = MaxPool2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters=self.layer_size[1], kernel_size=(3, 3), activation=self.activation,
                       kernel_initializer=self.initializer, name='conv2')(maxpool1)
        conv2 = BatchNormalization()(conv2)
        maxpool2 = MaxPool2D(pool_size=(2, 2))(conv2)
        flat = Flatten()(maxpool2)
        dense = Dense(units=self.layer_size[2], activation=self.activation, kernel_initializer=self.initializer,
                      name='dense')(flat)
        dense = BatchNormalization()(dense)
        output = Dense(units=self.layer_size[3], activation='sigmoid', kernel_initializer=self.initializer,
                       name='output')(dense)
        model = Model(inputs, output)
        model.summary()
        return model


if __name__ == '__main__':
    cnn = Cnn()
#    plot_model(cnn.model, to_file='cnn_model.png')
