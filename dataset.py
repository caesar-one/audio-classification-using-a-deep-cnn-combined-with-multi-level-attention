from typing import Tuple
from typing import List

import torch

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import numpy as np
import pandas as pd
import librosa
import librosa.display
from glob import glob
import pickle
import matplotlib.pyplot as plt

from params import *
from torchvggish.vggish_input import wavfile_to_examples

'''
This module is used to load the dataset.

Usage:

from dataset import load
X_train, X_val, X_test, y_train, y_val, y_test = load()
'''


def normalize(d: np.ndarray, _min: float, _max: float) -> np.ndarray:
    return (d - _min) / (_max - _min)


def load(save: bool = True,
         load_saved: bool = True,
         path: str = "",
         save_filename: str = "audio_data.pkl",
         cnn_type: str = "vggish",
         use_librosa: bool = False,
         overlap: bool = True,
         debug: bool = False,
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if overlap and T < 4:
        raise Exception('Number of slots must be >= 4')
    if not overlap and cnn_type != "resnet":
        raise Exception('This combination of params has not been implemented yet')

    if path + save_filename in glob(path + "*.pkl") and load_saved:
        # Load the dataset
        with open(path + save_filename, "rb") as f:
            (X_train, X_val, X_test, y_train, y_val, y_test) = pickle.load(f)
    else:
        # DATASET CREATION
        # We split the data into 3 sets: train (~60%), val (~20%), test (~20%).

        # Assign folders to the appropriate set
        wav_paths = glob(path + DATASET_PATH + "**/*.wav", recursive=True)
        wav_paths_train, wav_paths_val, wav_paths_test = [], [], []
        for p in wav_paths:
            if p.split("/")[-2] in ["fold1", "fold2"]:
                wav_paths_test.append(p)
            elif p.split("/")[-2] in ["fold3", "fold4"]:
                wav_paths_val.append(p)
            else:
                wav_paths_train.append(p)

        # Load the metadata
        metadata = pd.read_csv(path + METADATA_PATH)
        # Create a mapping from audio clip names to their respective label IDs
        name2class = dict(zip(metadata["slice_file_name"], metadata["classID"]))

        if cnn_type == "vggish":
            sr = SR_VGGISH
            samples_num = SAMPLES_NUM_VGGISH
            s_shape = S_VGGISH_SHAPE
            x_size = s_shape[0]
            y_size = s_shape[1]
        elif cnn_type == "resnet":
            sr = SR_RESNET
            samples_num = SAMPLES_NUM_RESNET
            s_shape = S_RESNET_SHAPE
            x_size = s_shape[1]
            y_size = s_shape[0]
        else:
            raise Exception("CNN type is not valid.")

        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
        for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
            for wav_path in tqdm(paths[:BATCH_SIZE] if debug else paths,
                                 desc=f"Converting {setname} samples in spectrograms"):
                # Create spectrogram and split into frames
                spec = create_spec(wav_path, cnn_type, sr, samples_num, x_size, y_size, use_librosa, overlap)
                # Split the spectrogram
                frames = split(spec, T, x_size, y_size, overlap)
                # Append each frames list to their respective set
                audio_filename = wav_path.split("/")[-1]
                if setname == "train":
                    X_train.append(frames)
                    y_train.append(int(name2class[audio_filename]))
                elif setname == "val":
                    X_val.append(frames)
                    y_val.append(int(name2class[audio_filename]))
                else:
                    X_test.append(frames)
                    y_test.append(int(name2class[audio_filename]))

        # Convert spectrogram lists into numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_tot = np.concatenate([X_train, X_val, X_test])

        # Normalize the dataset in between [0,1]
        _min = np.min(X_tot)
        _max = np.max(X_tot)
        X_train = normalize(X_train, _min, _max)
        X_val = normalize(X_val, _min, _max)
        X_test = normalize(X_test, _min, _max)

        if save:
            # Save the dataset
            with open(path + save_filename, "wb") as f:
                # protocol=4 saves correctly variables of size more than 4 GB
                pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f, protocol=4)

    return X_train, X_val, X_test, y_train, y_val, y_test


def overlapping_split(spec: np.ndarray, num_frames: int, frame_length: int) -> List[np.ndarray]:
    step_length = spec.shape[1] - frame_length

    '''
    window_length = s_shape[1]
    step_length = spec.shape[1] - s_shape[0]
    shape = (num_frames, window_length) + spec.shape[1:]
    strides = (spec.strides[1] * step_length,) + spec.strides
    return np.lib.stride_tricks.as_strided(spec, shape=shape, strides=strides)
    '''

    return [spec[:, i: i + frame_length]
            for i in range(0, spec.shape[1], step_length // (num_frames - 1))][:num_frames]


def contiguous_split(spec: np.ndarray, num_frames: int) -> List[np.ndarray]:
    frames = np.array_split(spec, num_frames, axis=1)
    for i in range(len(frames)):
        if len(frames[i]) != spec.shape[1] // num_frames:
            frames[i] = frames[i][:, :-1]
    return frames


def split(spec: np.ndarray, num_frames: int, x_size: int, y_size: int, overlap: bool):
    """
        :param spec: the spectogram, np.ndarray (s_shape[0], *), where first dim is freq and second dim is time
            (* means that the second dimension is variable)
        :param num_frames: number of pieces into which split the spectrogram
        :param s_shape: see params.py
        :param overlap: If True, the split has *num_frames* overlapping frame. Each frame has a fixed length of 1 second.
                         Otherwise, the split has T contiguous slots.
                         Each slot has length *samples_num* / *num_frames*
        :return: the spectrogram split in frames, np.ndarray (s_shape[0], s_shape[1])
    """
    if overlap:
        frames = overlapping_split(spec, num_frames, x_size)
    else:
        frames = contiguous_split(spec, num_frames)

    # Check shape and add a new axis in position 0 (this will be the channel axis)
    for i in range(len(frames)):
        assert frames[i].shape == (y_size, x_size)
        frames[i] = frames[i][np.newaxis, :, :]

    return frames


def create_spec(wav_path: str,
                cnn_type: str,
                sr: int,
                samples_num: int,
                x_size: int,
                y_size: int,
                use_librosa: bool,
                overlap: bool
                ) -> np.ndarray:
    """
    :param wav_path: the audio filename
    :param cnn_type: "resnet" or "vggish"
    :param sr: sampling rate,
    :param samples_num: number of total samples in the audio array (required for cutting/padding)
    :param s_shape: shape of a spectogram frame (see the function frame)
    :param use_librosa: If True, use librosa to generate the spectrogram, otherwise use torchvggish
    :param overlap: If True, the split has *num_frames* overlapping frame. Each frame has a fixed length of 1 second.
                         Otherwise, the split has T contiguous slots.
                         Each slot has length *samples_num* / *num_frames*
    :return: the spectrogram related to the audio data contained in *data*, np.ndarray (s_shape[0], *)
    """

    if use_librosa:
        # Load the audio clip stored at *wav_path* in an audio array
        audio_array, _ = librosa.load(wav_path, sr=sr)
        # Make sure that all audio arrays are of the same length *samples_num*
        # (cut if larger, zero-fill if smaller)
        audio_array = audio_array[:samples_num]
        reshaped_array = np.zeros((samples_num,))
        reshaped_array[:audio_array.shape[0]] = audio_array
        # Compute the spectrogram of the reshaped array.
        # The granularity on frequency (n_mels) axis is y_size
        # The granularity on time axis (hop_length) depends on whether we will divide the result in frames or not
        if cnn_type == "vggish":
            spec = librosa.feature.melspectrogram(reshaped_array, sr=sr, n_mels=64, hop_length=160, center=False,
                                                  htk=True, fmin=125, fmax=7500)
        elif cnn_type == "resnet":
            spec = librosa.feature.melspectrogram(reshaped_array, sr=sr, n_mels=y_size,
                                                  hop_length=samples_num // (x_size * 4) if overlap
                                                  else samples_num // (x_size * T))
        else:
            raise Exception("CNN type is not valid.")
        # Convert the spectrogram entries into decibels, with respect to ref=1.0.
        # If x is an entry of the spectrogram, this computes the scaling x ~= 10 * log10(x / ref)
        spec = librosa.power_to_db(spec)

    else:
        # wavfile_to_examples return spectrograms of shape (96, 64) grouped into a np.ndarray of shape (*, 96, 64)
        # * depends on the length of the audio file
        slots = wavfile_to_examples(wav_path, return_tensor=False)
        reshaped = np.zeros((4, slots.shape[1], slots.shape[2]))
        # reshaped.fill(-10)
        reshaped[:slots.shape[0]] = slots
        reshaped = np.swapaxes(reshaped, 1, 2)
        spec = np.concatenate(reshaped[:4], axis=1)

    return spec


def plot_spec(spec: np.ndarray, sr: int) -> None:
    librosa.display.specshow(spec, sr=sr)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # GENERATE DATASET

    '''
    data_path = '/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention-mechanism/'
    postfix = 'vggish_nospec.pkl'

    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load(
        load_saved=False,
        save=True,
        path=data_path,
        save_filename='audio_data_' + postfix,
        create_spec=False
    )
    '''

    # TRY WITH SINGLE WAV FILE

    # params
    cnn_type = 'vggish'
    wav_path = '/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention-mechanism/UrbanSound8K/audio/fold1/7061-6-0-0.wav'
    overlap = True
    T = 8

    if cnn_type == "vggish":
        sr = SR_VGGISH
        samples_num = SAMPLES_NUM_VGGISH
        s_shape = S_VGGISH_SHAPE
        x_size = s_shape[0]
        y_size = s_shape[1]
    elif cnn_type == "resnet":
        sr = SR_RESNET
        samples_num = SAMPLES_NUM_RESNET
        s_shape = S_RESNET_SHAPE
        x_size = s_shape[1]
        y_size = s_shape[0]
    else:
        raise Exception("CNN type is not valid.")

    ours = create_spec(wav_path, cnn_type, sr, samples_num, x_size, y_size, use_librosa=True, overlap=overlap)

    theirs = create_spec(wav_path, cnn_type, sr, samples_num, x_size, y_size, use_librosa=False, overlap=overlap)

    # Normalize the dataset in between [0,1]
    _min_our = np.min(ours)
    _max_our = np.max(ours)
    _min_their = np.min(theirs)
    _max_their = np.max(theirs)

    ours = normalize(ours, _min_our, _max_our)
    theirs = normalize(theirs, _min_their, _max_their)

    plot_spec(ours, sr)
    plot_spec(theirs, sr)
    if ours.shape == theirs.shape:
        plot_spec(ours - theirs, sr)
    else:
        print('our shape is different from their shape, {} and {}'.format(ours.shape, theirs.shape))

    ours = split(ours, T, x_size, y_size, overlap=overlap)
    theirs = split(theirs, T, x_size, y_size, overlap=overlap)

    for o in ours:
        plot_spec(o[0], sr)

    for t in theirs:
        plot_spec(t[0], sr)

