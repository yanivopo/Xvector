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
else:
    xvector_model.model.load_weights(os.path.join(model_dir_path, weight_name))

# xvector_model.evaluate_model(1, train_data_path)
xvector_model.load_embedded_model()
wave_file = ".\\data\\merge_wav\\merge1.wav"
wave_txt = ".\\data\\merge_wav\\merge1.txt"

with open(wave_txt, "r") as f_read:
    picks_real_msec = f_read.readlines()

mse = xvector_model.mse_sliding_window(wave_file)
plt.plot(mse)
picks_predict_sr, picks_predict_msec, picks_predict_10msec, mse_smos_correct, peak_points, mse = data_process_util.find_peaks(mse, picks_real_msec, window=8, treshold=0.8)
picks_real_10msec = [data_process_util.x_2_p(int(picks_real_msec[0].rstrip())), data_process_util.x_2_p(int(picks_real_msec[1].rstrip()))]


picks_predict_sr.insert(0, 0)

y, sr = sf.read(wave_file)
picks_predict_sr.append(len(y))
partition_list = []
partition_size = len(picks_predict_sr) - 1
embedded_xvector_examples = np.zeros(shape=(partition_size, xvector_model.layer_size[-1]))
for i in range(partition_size):
    part_wav = y[picks_predict_sr[i]:picks_predict_sr[i+1]]
    if len(part_wav) < 3 * sr:
        part_wav_suit = data_process_util.complete_sound(part_wav, sr)
    else:
        part_wav_suit = data_process_util.truncate_sound(part_wav, sr)
    fft_part = get_fft_spectrum(part_wav_suit)
    fft_part = fft_part[np.newaxis, :, :, np.newaxis]
    part_embedded_xvector = xvector_model.predict_embedded(fft_part)

    partition_list.append([picks_predict_sr[i], picks_predict_sr[i+1]])
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
partition_in_second = [p / 16000 for p in picks_predict_sr]


labels_according_k_means = np.zeros(len(mse_smos_correct))
peak_points.append(len(mse))
peak_points_2 = peak_points.copy()
peak_points_2.insert(0, 0)
leb = list(k_mean_model.labels_)

for i in range(len(peak_points)):
    l = leb[i]
    if l == 1:
        print(peak_points_2[i])
        print(peak_points_2[i+1])
        labels_according_k_means[peak_points_2[i]:peak_points_2[i+1]] = 1

plt.figure()
plt.plot(mse)
plt.grid()
plt.xlabel('time [10ms]')
plt.ylabel('MSE')
plt.plot(labels_according_k_means+1)
plt.plot(mse_smos_correct, '-gX', markevery=picks_predict_10msec[:-1], ms=20)
plt.plot(mse_smos_correct, '-rD', markevery=picks_real_10msec)
plt.show()

print("The real changes between the speakers are {}".format(picks_real_10msec))
print("The predict changes between the speakers are {}".format(picks_predict_10msec))




