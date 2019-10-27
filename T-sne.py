import os
from Xvector.model import Xvector
from matplotlib import pyplot as plt
from Xvector.wave_reader import get_fft_spectrum
import numpy as np
import soundfile as sf
from sklearn.manifold import TSNE
from tqdm import tqdm

model_dir_path = './save_model'
weight_name = 'weights_full_model-improvement-41-0.79.hdf5'
model_name = 'model.json'
xvector_model = Xvector(epochs=1)
xvector_model.model.load_weights(os.path.join(model_dir_path, weight_name))
xvector_model.load_embedded_model()

tsne_data_path = 'D:\\dataset\\woxceleb\\train_split'
dir_list = os.listdir(tsne_data_path)
y_label = []
x_train = []
sample_per_speaker = 100
number_of_speaker = 5
for i, j in tqdm(enumerate(dir_list[:number_of_speaker])):
    full_dir_name = os.path.join(tsne_data_path, j)
    sub_dir_list = os.listdir(full_dir_name)
    for k in sub_dir_list[:sample_per_speaker]:
        full_path_file = os.path.join(full_dir_name, k)
        y, _ = sf.read(full_path_file)
        fft_part_1 = get_fft_spectrum(y)
        fft_part_1 = fft_part_1[np.newaxis, :, :, np.newaxis]
        s1 = xvector_model.predict_embedded(fft_part_1)
        y_label.append(i)
        x_train.append(s1)
y_label = np.array(y_label)
x_train = np.array(x_train).reshape(-1, x_train[0].shape[-1])
X_embedded = TSNE(n_components=2, perplexity=20, n_iter=1400).fit_transform(x_train)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_label, cmap=plt.cm.get_cmap('Paired'))
plt.title("T-sne without training, number of speaker {} , number of sample {}".format(number_of_speaker, sample_per_speaker))
plt.savefig('./pictures/t_sne_{}_speaker'.format(number_of_speaker))
plt.show()

plt.figure()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_embed = pca.fit_transform(x_train)
explained_variance = np.round(np.sum(pca.explained_variance_ratio_), 2)
plt.scatter(pca_embed[:, 0], pca_embed[:, 1], c=y_label, cmap=plt.cm.get_cmap('Paired'))
plt.title("PCA training, number of speaker {}  explained_variance {:.2f} "
          .format(number_of_speaker, explained_variance))
plt.savefig('./pictures/PCA_{}_speaker'.format(number_of_speaker))
plt.show()