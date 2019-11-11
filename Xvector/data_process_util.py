import os
import numpy as np
from Xvector import wave_reader
import concurrent.futures
import soundfile as sf
import shutil
from pydub import AudioSegment
import argparse
from matplotlib import pyplot as plt

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
    y1_len_ms = len(y1) * 1 / sr1 * 1000
    y2_len_ms = len(y2) * 1 / sr2 * 1000
    point_of_change_1 = np.random.randint(low=0, high=y1_len_ms, size=1)[0]
    file1 = AudioSegment.from_wav(file1_path)
    file2 = AudioSegment.from_wav(file2_path)
    if point_of_change_1 < 6000:
        file1_part1 = file1 + file1[0:point_of_change_1]
        point_of_change_1_new = int(point_of_change_1 + y1_len_ms)
    else:
        file1_part1 = file1[0:point_of_change_1]
        point_of_change_1_new = point_of_change_1
    if point_of_change_1 > y1_len_ms - 6000:
        file1_part2 = file1 + file1[point_of_change_1:y1_len_ms]
    else:
        file1_part2 = file1[point_of_change_1:y1_len_ms]
    point_of_change_2 = int(point_of_change_1_new + y2_len_ms)

    combined_sounds = file1_part1 + file2 + file1_part2
    combined_sounds.export(new_file_name + '.wav', format="wav")
    change_point_list = [point_of_change_1_new, point_of_change_2]
    with open(new_file_name + '.txt', 'w') as f:
        for item in change_point_list:
            f.write("%s\n" % item)
    return change_point_list


def find_peaks(mse, picks_real_msec, window=10, treshold=0.7):
    weights = np.repeat(1.0, window) / window
    mse_smos = np.convolve(mse, weights, 'valid')
    mse_smos_norm = mse_smos / max(mse_smos)
    #    plt.figure()
    #    plt.plot(mse_smos_norm)

    if_big_treshold = mse_smos_norm > treshold
    index_big_treshold = np.where(if_big_treshold)[0]

    start_inx = []
    end_inx = []
    flag = 1
    for i in range(len(index_big_treshold)):
        if flag:
            start_inx.append(index_big_treshold[i])
        if i != (len(index_big_treshold) - 1):
            if index_big_treshold[i] + 1 == index_big_treshold[i + 1]:
                flag = 0
            else:
                end_inx.append(index_big_treshold[i])
                flag = 1
        else:
            end_inx.append(index_big_treshold[i])

    peak_points = []
    for i in range(len(start_inx)):
        peak_points.append(start_inx[i] + np.argmax(mse_smos[start_inx[i]:end_inx[i] + 1]) + int(window / 2))
    #    plt.figure()
    #    plt.plot(mse)
    #    plt.grid()
    mse_smos_correct = np.roll(mse_smos, int(window / 2))
    # mse_smos_correct = mse_smos
    #    plt.plot(mse_smos_correct, '-rD', markevery=peak_points)
    peak_points_sr = p_2_x(np.array(peak_points)) * 16
    peak_points_msec = p_2_x(np.array(peak_points))
    peak_points_sr.tolist()
    peak_points_msec.tolist()
    peak_points_sr = list(map(int, peak_points_sr))
    peak_points_msec = list(map(int, peak_points_msec))
    return peak_points_sr, peak_points_msec, peak_points, mse_smos_correct, peak_points, mse


def x_2_p(x):
    p = (x - 3000) // 100
    return p


def p_2_x(p):
    x = ((p + 1) / 10 - 3) * 1000 + 6000
    return x


def complete_sound(sound, sr, second=3):
    number_of_copy = (second * sr // len(sound)) + 1
    repeat_sound = np.repeat(sound, number_of_copy)
    return truncate_sound(repeat_sound, sr, second)


def truncate_sound(sound, sr, second=3):
    return sound[:sr * second]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example with long option names')
    parser.add_argument('-f', '--first_file', required=True, help="path to first audio")
    parser.add_argument('-s', '--second_file',required=True, help="path to second audio" )
    parser.add_argument('-o', '--output_dir', required=True, help ="dir for output merge file")
    parser.add_argument('-n', '--name_output_file', default='new_merge', help="dir for output merge file")

    args = parser.parse_args()

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
    file_1 = args.first_file   # 'D:\\dataset\\woxceleb\\14.wav'
    file_2 = args.second_file  # 'D:\\dataset\\woxceleb\\6.wav'
    output_dir = args.output_dir  # "C:\\Users\\USER\\Desktop\\Master\\Xvector\\data\\merge_wav\\"
    out_merge_file = args.name_output_file  # 'merge4'
    merge_wav_files(file_1, file_2, output_dir + out_merge_file)
