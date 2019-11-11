# Triplet Network for Speaker Diarization
Speaker diarization is the process of partitioning an input audio
stream into homogeneous segments according to the speaker identity. It answers the question “who spoke when” in a multi-speaker
environment.

###  To install with anaconda: 
```
conda create -n xvector python=3.6 
conda activate xvector 
pip install -r /path/to/requirements.txt

``` 
### To Run example: 
``` 
python main.py
``` 

### To create a new test example 
``` 
python /path/to/Xvector/data_procss_util.py -f first_audio.wav -s second_audio.wav -o path\to\output_dir\
```
Explanation of the diractories and files:

main.py - The main file.

Xvactor - The package for the Triplet Network for Speaker Diarization

Voxceleb_EDA_Notebook.ipynb - Notebook for the EDA

Data - sample of batches that we created from the real date. The original data can be found in "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/" (30 GB). Utilities for the pre-processing of the data located in "Xvector/data_process_util.py"

Data/merge_files/ - the merge wav files for the test.

save_model - contain traning model + weights