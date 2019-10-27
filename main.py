import os
from Xvector.model import Xvector
from matplotlib import pyplot as plt
from Xvector.data_generator import DataGeneratorLoad
from Xvector import data_process_util
import soundfile as sf
from Xvector.wave_reader import get_fft_spectrum
import numpy as np
from sklearn.cluster import KMeans
from pydub import AudioSegment

DO_TRAINING = False
model_dir_path = './save_model'
output_dir_path = './output'
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

# xvector_model.evaluate_model(1, train_data_path)
xvector_model.load_embedded_model()
wave_file = ".\\data\\merge_wav\\merge.wav"
wave_txt = ".\\data\\merge_wav\\merge.txt"

mse = xvector_model.mse_sliding_window(wave_file)
plt.plot(mse)
picks_predict = data_process_util.find_peaks(mse, window=6, treshold=0.7)

picks_predict.insert(0, 0)

y, sr = sf.read(wave_file)
picks_predict.append(len(y))
partition_list = []
partition_size = len(picks_predict) - 1
embedded_xvector_examples = np.zeros(shape=(partition_size, xvector_model.layer_size[-1]))
for i in range(partition_size):
    part_wav = y[picks_predict[i]:picks_predict[i+1]]
    if len(part_wav) < 3 * sr:
        part_wav_suit = data_process_util.complete_sound(part_wav, sr)
    else:
        part_wav_suit = data_process_util.truncate_sound(part_wav, sr)
    fft_part = get_fft_spectrum(part_wav_suit)
    fft_part = fft_part[np.newaxis, :, :, np.newaxis]
    part_embedded_xvector = xvector_model.predict_embedded(fft_part)

    partition_list.append([picks_predict[i], picks_predict[i+1]])
    embedded_xvector_examples[i] = part_embedded_xvector


def k_means(x_train, n_class=2, n_init=10):
    k_mean = KMeans(n_clusters=n_class, n_init=n_init)
    km_model = k_mean.fit(x_train)
    return km_model


def resegmentation_to_different_speaker(wave_file, partition_array, label, number_of_speaker = 2):
    file1 = AudioSegment.from_wav(wave_file)
    for i in range(number_of_speaker):
        i_speaker_partiton = partition_array[np.where(label == i)]
        for ii in range(i_speaker_partiton.shape[0]):
            if ii == 0:
                part_sounds = file1[int(i_speaker_partiton[ii][0] * 0.0625):int(i_speaker_partiton[ii][1] * 0.0625)]
            else:
                part_sounds += file1[int(i_speaker_partiton[ii][0] * 0.0625):int(i_speaker_partiton[ii][1] * 0.0625)]
        part_sounds.export(output_dir_path + '/part_' + str(i) + '.wav', format="wav")


number_of_speaker = 2
k_mean_model = k_means(embedded_xvector_examples, number_of_speaker)
partition_array = np.array(partition_list)
label = k_mean_model.labels_
resegmentation_to_different_speaker(wave_file, partition_array, label, number_of_speaker)





