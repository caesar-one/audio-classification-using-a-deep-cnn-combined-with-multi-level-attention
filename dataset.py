import numpy as np
import pandas as pd
import librosa.display
from glob import glob
from tqdm import tqdm
import pickle

'''
This module is used to preprocess and load the dataset.
Usage: 
from dataset import load
X_train, X_test, y_train, y_test = load()
'''

dataset_path = "UrbanSound8K/audio"
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
samples_number = 88200

def load(convert_to_log_scale = False, save = True, load_saved = True):
    if len(glob("*.pkl")) and load_saved:
        with open("audio_X_train.pkl", "rb") as f: X_train = pickle.load(f)
        with open("audio_y_train.pkl", "rb") as f: y_train = pickle.load(f)
        with open("audio_X_test.pkl", "rb") as f: X_test = pickle.load(f)
        with open("audio_y_test.pkl", "rb") as f: y_test = pickle.load(f)
    else:
        wav_paths = glob(dataset_path + "/**/*.wav", recursive=True)
        wav_paths_train, wav_paths_test = [], []
        for p in wav_paths:
            if p.split("/")[-2] in ["fold1","fold2"]: wav_paths_test.append(p)
            else: wav_paths_train.append(p)
        metadata = pd.read_csv(metadata_path)
        name2class = dict(zip(metadata["slice_file_name"],metadata["class"]))
        X_train, y_train, X_test, y_test = [], [], [], []
        for paths,setname in zip([wav_paths_train, wav_paths_test],["train","test"]):
            for w in tqdm(paths, desc = "Converting songs in spectrograms"):
                song, sr = librosa.load(w)
                song = song[:samples_number]
                reshaped_song = np.zeros((samples_number,))
                reshaped_song[:song.shape[0]]=song
                spectrogram = librosa.feature.melspectrogram(reshaped_song)
                if convert_to_log_scale: spectrogram = librosa.power_to_db(spectrogram)
                song_filename = w.split("/")[-1]
                if setname == "train":
                    X_train.append(spectrogram)
                    y_train.append(name2class[song_filename])
                else:
                    X_test.append(spectrogram)
                    y_test.append(name2class[song_filename])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        if save:
            with open("audio_X_train.pkl","wb") as f: pickle.dump(X_train,f)
            with open("audio_y_train.pkl","wb") as f: pickle.dump(y_train,f)
            with open("audio_X_test.pkl","wb") as f: pickle.dump(X_test,f)
            with open("audio_y_test.pkl","wb") as f: pickle.dump(y_test,f)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load(convert_to_log_scale = False, save = True, load_saved = True)