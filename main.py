import os
from Xvector.model import Xvector
from matplotlib import pyplot as plt
from Xvector.data_generator import DataGeneratorLoad
DO_TRAINING = False
data_dim = (512, 299)
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
# json_file = open(os.path.join(model_dir_path, model_name), 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
else:
    xvector_model.model.load_weights(os.path.join(model_dir_path, weight_name))
xvector_model.evaluate_model(1, train_data_path)
xvector_model.load_embedded_model()
wave_file = ".\\data\\merge_wav\\merge.wav"
mse = xvector_model.mse_sliding_window(wave_file)
plt.plot(mse)
plt.show()




