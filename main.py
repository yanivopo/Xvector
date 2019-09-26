import os
from Xvector.model import Xvector
from matplotlib import pyplot as plt
from Xvector.data_generator import DataGeneratorLoad
from Xvector import data_process_util
import soundfile as sf
from Xvector.wave_reader import get_fft_spectrum
import numpy as np
from sklearn.cluster import KMeans

DO_TRAINING = False
model_dir_path = './save_model'
weight_name = 'weights_full_model-improvement-41-0.79.hdf5'
model_name = 'model.json'
train_data_path = 'D:\\dataset\\woxceleb\\temp_train'
valid_data_path = 'D:\\dataset\\woxceleb\\temp_valid'
xvector_model = Xvector(epochs=1)
if DO_TRAINING:
    training_generator = DataGeneratorLoad(data_dir_name=train_data_path, step_per_epoch=300)
    valid_generator = DataGeneratorLoad(data_dir_name=valid_data_path, step_per_epoch=10)
    xvector_model.fit(training_generator, valid_generator, save_model=False)
    pass
else:
    xvector_model.model.load_weights(os.path.join(model_dir_path, weight_name))

#xvector_model.evaluate_model(1, train_data_path)
xvector_model.load_embedded_model()
wave_file = ".\\data\\merge_wav\\merge.wav"
wave_txt = ".\\data\\merge_wav\\merge.txt"

mse = xvector_model.mse_sliding_window(wave_file)
plt.plot(mse)
plt.show()
