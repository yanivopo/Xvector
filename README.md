# Triplet Network for Speaker Diarization
Xvactor - package using Triplet Network for Speaker Diarization
Speaker diarization is the process of partitioning an input audio
stream into homogeneous segments according to the speaker identity. It answers the question “who spoke when” in a multi-speaker
environment.

![triplet](https://miro.medium.com/max/1604/0*AX2TSZNk19_gDgTN.png "Triplet loss equation")

![principle](https://miro.medium.com/max/602/0*_WNBFcRVEOz6QM7R. "Using triplet loss in learning")
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

* `main.py` - The main file of the repo.

 * Voxceleb_EDA_Notebook.ipynb - Notebook presenting data EDA,you can also have a look at `VoxCeleb_EDA.html` for interactive data insights (open in browser)

 * Data - sample of batches we created from real data. The original data can be found in `http://www.robots.ox.ac.uk/~vgg/data/voxceleb` (30 GB). Utilities for pre-processing the data located in `Xvector/data_process_util.py`

`Data/merge_files/` - contains merge wav files for testing.

`save_model` - contain trained model + weights