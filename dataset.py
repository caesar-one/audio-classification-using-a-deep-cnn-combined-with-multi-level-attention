import numpy as np
import pandas as pd
import librosa.display
from glob import glob
from tqdm import tqdm
import pickle
import torch

'''
This module is used to preprocess and load the dataset.
Usage: 
from dataset import load
X_train, X_test, y_train, y_test = load()
'''

dataset_path = "UrbanSound8K/audio"
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
save_filename = "audio_data.pkl"
samples_number = 88200
s_size = 224

def normalize(d):
    _min, _max = np.min(d), np.max(d)
    result = (d - _min) / (_max - _min)
    return result

def load(save = True, load_saved = True, slots_num=4, use_sliding=None):
    if save_filename in glob("*.pkl") and load_saved:
        with open(save_filename, "rb") as f:
            (X_train, X_val, X_test, y_train, y_val, y_test) = pickle.load(f)
    else:
        wav_paths = glob(dataset_path + "/**/*.wav", recursive=True)
        wav_paths_train, wav_paths_val, wav_paths_test = [], [], []
        for p in wav_paths:
            if p.split("/")[-2] in ["fold1","fold2"]: wav_paths_test.append(p)
            if p.split("/")[-2] in ["fold3","fold4"]: wav_paths_val.append(p)
            else: wav_paths_train.append(p)
        metadata = pd.read_csv(metadata_path)
        name2class = dict(zip(metadata["slice_file_name"],metadata["classID"]))
        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
        for paths,setname in zip([wav_paths_train, wav_paths_val, wav_paths_test],["train", "val", "test"]):
            for w in tqdm(paths, desc = f"Converting {setname} samples in spectrograms"):
                sample, sr = librosa.load(w)
                sample = sample[:samples_number]
                reshaped_sample = np.zeros((samples_number,))
                reshaped_sample[:sample.shape[0]]=sample
                # Division multiple time pieces
                slots = np.split(reshaped_sample, slots_num)
                spectrograms = [librosa.feature.melspectrogram(s, hop_length=samples_number // (s_size * slots_num),
                                                               n_mels=s_size) for s in slots]
                spectrograms = [librosa.power_to_db(s) for s in spectrograms]
                # spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(reshaped_sample)), ref=np.max)
                spectrograms = [s[None, :s_size, :s_size] for s in spectrograms]
                spectrograms = normalize(spectrograms)
                #spectrograms = [preprocess(s) for s in spectrograms] # normalize spectrogram according to pretrained model requirements
                sample_filename = w.split("/")[-1]
                if setname == "train":
                    X_train.append(spectrograms)
                    y_train.append(int(name2class[sample_filename]))
                elif setname == "val":
                    X_val.append(spectrograms)
                    y_val.append(int(name2class[sample_filename]))
                else:
                    X_test.append(spectrograms)
                    y_test.append(int(name2class[sample_filename]))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        if save:
            with open(save_filename, "wb") as f:
                pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test),f,protocol=4)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load(save = True, load_saved = False)

    #for e in tqdm(X_train, desc='Preprocess'):
        #preprocess(e)
