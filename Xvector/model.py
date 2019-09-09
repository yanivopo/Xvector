from keras.layers import Input, concatenate, Reshape
from keras import Model
from Xvector.cnn_model import Cnn
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import soundfile as sf
from Xvector.wave_reader import get_fft_spectrum
import numpy as np
from tqdm import tqdm
import pickle
import os


def triplet_loss(y_true, y_pred):
    query = y_pred[:, 0]
    pos = y_pred[:, 1]
    neg = y_pred[:, 2]
    triplet_loss_out = tf.reduce_sum(tf.maximum(1 - tf.norm(query - neg, axis=1) + tf.norm(query - pos, axis=1), 0))
    return triplet_loss_out


class Xvector():
    def __init__(self, data_dim=(512, 299), optimizer='adam', layer_size=[10, 16, 20, 100], epochs=30):
        self.data_dim = data_dim
        self.layer_size = layer_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.model = self.create_model()
        self.embed_model = None

    def create_model(self):
        conv_model = Cnn(self.data_dim)
        inputs_q = Input(shape=(*self.data_dim, 1), name='q_input')
        inputs_p = Input(shape=(*self.data_dim, 1), name='p_input')
        inputs_n = Input(shape=(*self.data_dim, 1), name='n_input')
        q_vec_out = conv_model.model(inputs_q)
        p_vec_out = conv_model.model(inputs_p)
        n_vec_out = conv_model.model(inputs_n)
        q_vec_out = Reshape((1, self.layer_size[-1]))(q_vec_out)   #to check tf,newaxis
        p_vec_out = Reshape((1, self.layer_size[-1]))(p_vec_out)
        n_vec_out = Reshape((1, self.layer_size[-1]))(n_vec_out)
        out = concatenate([q_vec_out, p_vec_out, n_vec_out], axis=1, name='output')
        model = Model(inputs=[inputs_q, inputs_p, inputs_n], outputs=out)
        model.summary()
        return model

    def fit(self, training_generator, valid_generator,  save_model=True, save_model_name='temp'):
        self.model.compile(optimizer=self.optimizer, loss=triplet_loss)
        callbacks_list = []
        if save_model:
            filepath = save_model_name + "_weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
            model_check_point = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True)
            callbacks_list.append(model_check_point)

        history = self.model.fit_generator(generator=training_generator, epochs=self.epochs,
                                           validation_data=valid_generator, use_multiprocessing=False,
                                           workers=6, callbacks=callbacks_list)
        self.load_embedded_model()
        return history

    def evaluate_model(self, number_of_batch=1, dir_name='temp_train', top_of_k=5):
        a = np.empty((number_of_batch * 16, 3, 100))

        index_of_file = np.random.choice(np.arange(100), number_of_batch)
        for i in range(number_of_batch):
            file_name = str(index_of_file[i]) + ".pkl"
            with open(os.path.join(dir_name, file_name), 'rb') as handle:
                s = pickle.load(handle)[0]
            a[i * 16:(i + 1) * 16] = self.model.predict(s)
        p_predict = a[:, 1]
        q_predict = a[:, 0]
        p_predict = p_predict[:, np.newaxis, :]
        error = p_predict - q_predict
        error = np.square(error)
        mse = np.sum(error, axis=-1)
        sort_index_of_min_error = np.argsort(mse, axis=0)
        count = 0
        for i in range(sort_index_of_min_error.shape[1]):
            if i in sort_index_of_min_error[:top_of_k, i]:
                count += 1
        print("The percentage of sample in the top {} is {}".format(top_of_k,
                                                                    100 * (count / sort_index_of_min_error.shape[1])))
        return sort_index_of_min_error, mse
        pass

    def load_embedded_model(self):
        inputs_q = Input(shape=(*self.data_dim, 1), name='q_input')
        out = self.model.get_layer('model_1')(inputs_q)
        new_model = Model(inputs=inputs_q, outputs=out)
        self.embed_model = new_model

    def predict_embedded(self, data):
        return self.embed_model.predict(data)

    def mse_sliding_window(self, wave_file):
        y, sr = sf.read(wave_file)
        N = sr * 3     # 3 second the size of windows
        mil_sec = sr // 10
        mse = []
        for i in tqdm(range((len(y) - 2 * N) // mil_sec)):
            part_1 = y[i * mil_sec:i * mil_sec + N]
            part_2 = y[i * mil_sec + N:i * mil_sec + 2 * N]
            fft_part_1 = get_fft_spectrum(part_1)
            fft_part_2 = get_fft_spectrum(part_2)
            fft_part_1 = fft_part_1[np.newaxis, :, :, np.newaxis]
            fft_part_2 = fft_part_2[np.newaxis, :, :, np.newaxis]
            s1 = self.predict_embedded(fft_part_1)
            s2 = self.predict_embedded(fft_part_2)
            mse.append(np.linalg.norm(s1 - s2))
        return np.array(mse)
