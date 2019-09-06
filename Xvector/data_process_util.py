import os
import numpy as np
from Xvector import wave_reader
import concurrent.futures
import soundfile as sf
import shutil
from pydub import AudioSegment

output_path = 'D:\\dataset\\woxceleb\\temp_yaniv'
input_dir_path = 'D:\\dataset\\woxceleb\\train_split'


def stft_transformation(lt, input_dir=input_dir_path, output=output_path):
    os.mkdir(os.path.join(output, lt))
    print(lt)
    full_dir_name = os.path.join(input_dir, lt)
    my_sub_list = os.listdir(full_dir_name)
    count = 0
    for j in my_sub_list:
        full_file_name = os.path.join(full_dir_name, j)
        stft = wave_reader.get_fft_spectrum(full_file_name)
        np.save(output+'\\' + lt + '\\'+str(count), stft)
        count += 1


def split_file(file_path, dir_to_new_files, sec=3):
    file_name = os.path.basename(file_path).split('.')[0]
    y, sr = sf.read(file_path)
    N = sr*sec
    number_of_part = int((len(y) - N) / sr)
    for i in range(number_of_part):
        part_i = y[sr*i:(sr*i)+N]
        name = file_name + '_' + str(i) + '.wav'
        sf.write(os.path.join(dir_to_new_files, name), part_i, 16000)


def split_all(lt, output_dir=output_path, input_dir='D:\\dataset\\woxceleb\\yaniv_input'):
    os.mkdir(os.path.join(output_dir, lt))
    print(lt)
    full_dir_name = os.path.join(input_dir, lt)
    my_sub_list = os.listdir(full_dir_name)
    for d in my_sub_list:
        full_file_name = os.path.join(full_dir_name, d)
        split_file(full_file_name, output_dir+'\\' + lt)


def reduce_dir(lt, output_dir=output_path, input_dir='D:\\dataset\\woxceleb\\yaniv_input'):
    os.mkdir(os.path.join(output_dir, lt))
    full_dir_name = os.path.join(input_dir, lt)
    my_sub_list = os.listdir(full_dir_name)
    count = 0
    for i, j in enumerate(my_sub_list):
        full_file_name = os.path.join(full_dir_name, j)
        my_sub_file = os.listdir(full_file_name)
        for file_ in my_sub_file:
            full_file_name_path = os.path.join(full_file_name, file_)
            print(full_file_name_path)
            shutil.copy(full_file_name_path, output_dir+'\\' + lt + '\\'+str(count)+'.wav')
            count += 1


def merge_wav_files(file1_path, file2_path, new_file_name='combined_sounds'):
    y1, sr1 = sf.read(file1_path)
    y2, sr2 = sf.read(file2_path)
    y1_len_ms = len(y1)*1/sr1*1000
    y2_len_ms = len(y2)*1/sr2*1000
    max_len = min(y1_len_ms, y2_len_ms)
    point_of_change = np.random.randint(low =6000 , high=max_len,size=1)
    file1 = AudioSegment.from_wav(file1_path)
    file2 = AudioSegment.from_wav(file2_path)
    file1_part1 = file1[0:point_of_change]
    file1_part2 = file1[point_of_change:y1_len_ms]
    combined_sounds = file1_part1 + file2 + file1_part2
    combined_sounds.export(new_file_name + '.wav', format="wav")
    file_label = np.zeros([1, int(y1_len_ms+y2_len_ms)])
    file_label[0, int(point_of_change):int(point_of_change + int(y2_len_ms))] = np.ones([1,int(y2_len_ms)])
    change_point_list = [point_of_change]
    with open(new_file_name+'.txt', 'w') as f:
        for item in change_point_list:
            f.write("%s\n" % item)
    return file_label, point_of_change


if __name__ == '__main__':

    # input_dir = 'D:\\dataset\\woxceleb\\train_split'
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     dir_list = os.listdir(input_dir)
    #     dir_list = dir_list[:10]
    #     executor.map(stft_transformation, dir_list)    # methods to execute calls asynchronously
    # reduce_dir('id10001')
    # input_dir = 'D:\\dataset\\woxceleb\\yaniv_input'
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     my_list = os.listdir(input_dir)
    #     executor.map(reduce_dir, my_list)
    # split_all('id10001')
    # input_dir = 'D:\\dataset\\woxceleb\\yaniv_input'
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     my_list = os.listdir(input_dir)
    #     executor.map(split_all, my_list)
    file_1 = 'D:\\dataset\\woxceleb\\8.wav'
    file_2 = 'D:\\dataset\\woxceleb\\14.wav'
    output_dir = "C:\\Users\\USER\\Desktop\\Master\\Xvector\\data\\merge_wav\\"
    out_merge_file = 'merge'
    merge_wav_files(file_1, file_2, output_dir + out_merge_file)
    pass




