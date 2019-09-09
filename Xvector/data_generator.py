# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:41:52 2019

@author: USER
"""
import numpy as np
import keras
import random
import os
import pickle


class DataGeneratorCreate(keras.utils.Sequence):
    def __init__(self, dir_name, batch_size=16, dim=(512, 299), n_channels=1, step_per_epoch=5000,
                 output_dir='./batch_train/'):
        self.dim = dim
        self.batch_size = batch_size
        self.dir_name = dir_name
        self.n_channels = n_channels
        self.output_dir = output_dir
        self.step_per_epoch = step_per_epoch

    def __batch_generation(self, arr_list, index):
        # Initialization
        batch = [np.empty((self.batch_size, *self.dim, self.n_channels)).astype(np.float16)]
        triple_batch = 3 * batch
        data_idx = np.random.choice(arr_list, size=(self.batch_size, 2))
        full_dir_name_one_idx = np.core.defchararray.add(self.dir_name + '\\', data_idx[:, 0])
        full_dir_name_sec_idx = np.core.defchararray.add(self.dir_name + '\\', data_idx[:, 1])

        for i in range(self.batch_size):
            file_one_idx = os.listdir(full_dir_name_one_idx[i])
            file_sec_idx = os.listdir(full_dir_name_sec_idx[i])
            choice_one_idx = random.sample(file_one_idx, k=2)
            choice_sec_idx = random.sample(file_sec_idx, k=1)
            q = np.load(os.path.join(full_dir_name_one_idx[i], choice_one_idx[0]))
            p = np.load(os.path.join(full_dir_name_one_idx[i], choice_one_idx[1]))
            n = np.load(os.path.join(full_dir_name_sec_idx[i], choice_sec_idx[0]))
            triple_batch[0][i, ] = q[:, :, np.newaxis]
            triple_batch[1][i, ] = p[:, :, np.newaxis]
            triple_batch[2][i, ] = n[:, :, np.newaxis]
        triplet_batch = ({'q_input': triple_batch[0], 'p_input': triple_batch[1], 'n_input': triple_batch[2]},
                         {'output': np.ones(self.batch_size)})
        with open(self.output_dir + str(index) + '.pkl', 'wb') as handle:
            pickle.dump(triplet_batch, handle)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.step_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        my_list = os.listdir(self.dir_name)
        arr_list = np.array(my_list)
        # Generate batch
        self.__batch_generation(arr_list, index)


class DataGeneratorLoad(keras.utils.Sequence):
    def __init__(self, step_per_epoch, data_dir_name):
        self.step_per_epoch = step_per_epoch
        self.data_dir_name = data_dir_name

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, index):
        # load batch
        batch = self.__data_generation_load()
        return batch

    def __data_generation_load(self, ):
        my_list = os.listdir(self.data_dir_name)
        rand_index = np.random.choice(len(my_list), 1)[0]
        with open(os.path.join(self.data_dir_name, my_list[rand_index]), 'rb') as handle:
            batch_load = pickle.load(handle)
        return batch_load


if __name__ == '__main__':
    params = {'dim': (512, 299),
              'batch_size': 16,
              'n_channels': 1}
    generator_create = DataGeneratorCreate("D:\\dataset\\woxceleb\\new_fft_valid", step_per_epoch=3,
                                           output_dir='C:\\Users\\USER\\Desktop\\Master\\Xvector\\data\\batch_valid\\', **params)
    generator_load = DataGeneratorLoad(step_per_epoch=1,
                                       data_dir_name='C:\\Users\\USER\\Desktop\\Master\\Xvector\\data\\batch_train\\')
