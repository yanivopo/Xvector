import os
import random
from glob import glob
from pydub import AudioSegment
import numpy as np
from progressbar import progressbar
from pydub.playback import play
from scipy.io import wavfile as wavf

SILANCE_LIKELYHOOD = .5
MIN_SIL_LEN = 250
MAX_SIL_LEN = 3000
MAX_SEG_LEN = 3000

class SyntheticMultiSpeakerGen:
    def __init__(self, path_to_vox, output_path, sample_len_sec=10):
        self.sample_len_sec = sample_len_sec
        self.output_path = output_path
        self.path_to_vox = path_to_vox
        self.all_files = glob(path_to_vox + "/**/*.wav", recursive=True)
        self.persons = list(filter(lambda x: os.path.isdir(os.path.join(path_to_vox, x)), os.listdir(path_to_vox)))
        self.person_files = {person: glob(path_to_vox + "/" + person + "/**/*.wav", recursive=True) for person in
                             self.persons}
        for dir in ["label", "wav"]:
            if not os.path.exists(f"{self.output_path}/{dir}"):
                os.mkdir(f"{self.output_path}/{dir}")

    def generate_samples(self, num_samples=2e6):
        [self.get_sample() for _ in progressbar(range(num_samples))]

    def get_sample(self):
        person_ids = random.sample(self.persons, 2)
        label_0, sample_0 = self.get_single_person_sample(person_ids[0])
        label_1, sample_1 = self.get_single_person_sample(person_ids[1])
        label = label_0 + label_1 * 2
        sample = sample_0.overlay(sample_1, 0)

        id = random.randint(0, 10000000)
        sample.export(f"{self.output_path}/wav/{person_ids[0]}_{person_ids[1]}_{id}.wav", format='wav')
        np.save(f"{self.output_path}/label/{person_ids[0]}_{person_ids[1]}_{id}.npy", label)

    def get_single_person_sample(self, person_id):
        sample = AudioSegment.silent(duration=self.sample_len_sec * 1000)
        label = np.zeros(len(sample))
        start_idx = 0
        while start_idx < len(sample):
            if random.random() > SILANCE_LIKELYHOOD:
                start_idx += random.randint(MIN_SIL_LEN, MAX_SIL_LEN)
            else:
                random_file = random.choice(self.person_files[person_id])
                new_seg = self.get_audio_segment(random_file)
                sample = sample.overlay(new_seg, position=start_idx)
                label[start_idx: start_idx + len(new_seg)] = 1
                start_idx += len(new_seg)
        return label, sample

    def get_audio_segment(self, random_file):
        seg = AudioSegment.from_wav(random_file)
        seg_len = int(len(seg))
        seg_middle = int(seg_len // 2 + random.randint(-seg_len // 2 + 100, seg_len // 2 - 100))
        seg = seg[
              random.randint(0, seg_middle - 100):random.randint(seg_middle + 100, seg_len)]
        seg = seg[:MAX_SEG_LEN]
        return seg


class SyntheticMultiSpeakerGenVer2:
    def __init__(self, path_to_vox, output_path, n_segments = 10):
        self.n_segments = n_segments
        self.output_path = output_path
        self.path_to_vox = path_to_vox
        self.all_files = glob(path_to_vox + "/**/*.wav", recursive=True)
        self.persons = list(filter(lambda x: os.path.isdir(os.path.join(path_to_vox, x)), os.listdir(path_to_vox)))
        self.person_files = {person: glob(path_to_vox + "/" + person + "/**/*.wav", recursive=True) for person in
                             self.persons}
        self.fs = 1000
        for dir in ["label", "wav"]:
            if not os.path.exists(f"{self.output_path}/{dir}"):
                os.mkdir(f"{self.output_path}/{dir}")

    def generate_samples(self, n_samples):
        [self.create_sample() for _ in range(n_samples)]

    def create_sample(self):
        sample = AudioSegment.empty()
        speaker_1, label_1, speaker_2, label_2, persons_id = self.get_signal_and_label()
        for i in range(self.n_segments):
            signal,signal_label = random.choice([[speaker_1,label_1],[speaker_2,label_2]])
            if i==0:
                segment_len = 3000
                segment_start = random.randint(0,len(signal) - segment_len)
                sample += signal[segment_start:segment_start+segment_len]
                label = signal_label[segment_start:segment_start+segment_len]
            else:
                if random.random() > 0.5:
                    segment_len = random.randint(0,len(signal))
                    segment_start = random.randint(0, len(signal) - segment_len)
                    sample += signal[segment_start:segment_start + segment_len]
                    label = np.append(label, signal_label[segment_start:segment_start + segment_len])
                else:
                    segment_len = random.randint(50,1000)
                    segment_start = random.randint(0, min(len(speaker_1),len(speaker_2)) - segment_len)
                    sample += speaker_1[segment_start:segment_start + segment_len]\
                        .overlay(speaker_2[segment_start:segment_start + segment_len], segment_start)
                    label = np.append(label,label_1[segment_start:segment_start + segment_len] \
                                      + label_2[segment_start:segment_start + segment_len])
        id = random.randint(0, 10000000)
        sample.export(f"{self.output_path}/wav/{persons_id[0]}_{persons_id[1]}_{id}.wav", format='wav')
        np.save(f"{self.output_path}/label/{persons_id[0]}_{persons_id[1]}_{id}.npy", label)

    def get_signal_and_label(self):
        person_curr = random.sample(self.persons, 2)
        speaker_file_1 = random.choice(self.person_files[person_curr[0]])
        speaker_1 = AudioSegment.from_file(speaker_file_1)
        label_1 = np.ones(len(speaker_1))
        speaker_file_2 = random.choice(self.person_files[person_curr[1]])
        speaker_2 = AudioSegment.from_file(speaker_file_2)
        label_2 = 2*np.ones(len(speaker_2))
        return speaker_1, label_1, speaker_2, label_2, person_curr

    def add_silent(self,speaker,label,start_idx,stop_idx):
        speaker[start_idx:stop_idx] = AudioSegment.silent(duration=(stop_idx-start_idx))
        label[start_idx:stop_idx] = 0
        return  speaker, label


if __name__ == '__main__':
    snyth_convo_gen = SyntheticMultiSpeakerGenVer2("/home/aviv/Data/Voxceleb/vox1_dev_wav/wav",
                                               "/home/aviv/Data/Voxceleb/synthetic")
    snyth_convo_gen.generate_samples(5000)
