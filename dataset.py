from typing import Tuple
from typing import List

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
import h5py

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


def loadbak(save: bool = True,
            load_saved: bool = True,
            path: str = "",
            filename: str = "audio_data.pkl",
            cnn_type: str = "vggish",
            use_librosa: bool = False,
            overlap: bool = True,
            debug: bool = False,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if overlap and T < 4:
        raise Exception('Number of slots must be >= 4')
    if not overlap and cnn_type == "vggish" and not use_librosa and T != 4:
        raise Exception('This combination of params needs T = 4')

    # Force to use librosa for resnet
    if cnn_type == "resnet" and not use_librosa:
        print('use_librosa forced to True')
        use_librosa = True

    if path + filename in glob(path + "*.pkl") and load_saved:
        # Load the dataset
        with open(path + filename, "rb") as f:
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
                # Add a new axis for the channel dimension
                for i in range(len(frames)):
                    frames[i] = frames[i][np.newaxis, :, :]
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

        # Plot frames for manual check on the file fold10/100795-3-0-0.wav (can be removed)
        if debug:
            for i in range(T):
                plot_spec(X_train[5, i, 0], sr)

        save_filename = f"audio_data_{cnn_type}{'' if use_librosa else '_native'}_{T}{'_s' if overlap else ''}"
        if save:
            # Save the dataset
            with open(path + save_filename, "wb") as f:
                # protocol=4 saves correctly variables of size more than 4 GB
                pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f, protocol=4)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load(save: bool = True,
         load_saved: bool = True,
         path: str = "",
         filename: str = "",
         cnn_type: str = "vggish",
         use_librosa: bool = False,
         overlap: bool = True,
         debug: bool = False,
         features: Tuple[str] = ("spectrogram", "mfcc", "crp")
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if overlap and T < MAX_SECONDS:
        raise Exception(f'Number of slots must be >= {MAX_SECONDS}')
    if not overlap and cnn_type == "vggish" and not use_librosa and T != MAX_SECONDS:
        raise Exception(f'This combination of params needs T = {MAX_SECONDS}')
    # Force to use librosa for resnet
    if cnn_type == "resnet" and not use_librosa:
        print('use_librosa forced to True')
        use_librosa = True

    if path + filename in glob(path + "*.pkl") and load_saved:
        # Load the dataset
        with open(path + filename, "rb") as f:
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

    #X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
    X_train = dataset.create_dataset("X_train", shape=(len(wav_paths_train),T,len(features),y_size,x_size))
    y_train = dataset.create_dataset("y_train", shape=(len(wav_paths_train),T,len(features),y_size,x_size))
    X_val = dataset.create_dataset("X_val", shape=(len(wav_paths_val),T,len(features),y_size,x_size))
    y_val = dataset.create_dataset("y_val", shape=(len(wav_paths_val),T,len(features),y_size,x_size))
    X_test = dataset.create_dataset("X_test", shape=(len(wav_paths_test),T,len(features),y_size,x_size))
    y_test = dataset.create_dataset("y_test", shape=(len(wav_paths_test),T,len(features),y_size,x_size))
    for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
        counter = 0
        for wav_path in tqdm(paths[:BATCH_SIZE] if debug else paths,
                             desc=f"Converting {setname} samples in spectrograms"):
            # Create spectrogram and split into frames
            spec = create_spec(wav_path, cnn_type, sr, samples_num, x_size, y_size, use_librosa, overlap)
            # Split the spectrogram
            frames = split(spec, T, x_size, y_size, overlap)
            # Add a new axis for the channel dimension
            for i in range(len(frames)):
                frames[i] = frames[i][np.newaxis, :, :]
            # Append each frames list to their respective set
            audio_filename = wav_path.split("/")[-1]

            if setname == "train":
                X_train[counter] = frames
                y_train[counter] = int(name2class[audio_filename])
            elif setname == "val":
                X_val[counter] = frames
                y_val[counter] = int(name2class[audio_filename])
            else:
                X_test[counter] = frames
                y_test[counter] = int(name2class[audio_filename])
            counter += 1

        # Convert spectrogram lists into numpy arrays
        #X_train = np.array(X_train)
        #y_train = np.array(y_train)
        #X_val = np.array(X_val)
        #y_val = np.array(y_val)
        #X_test = np.array(X_test)
        #y_test = np.array(y_test)

        #X_tot = np.concatenate([X_train, X_val, X_test])

        # Normalize the dataset in between [0,1]
        #_min = np.min(X_tot)
        #_max = np.max(X_tot)
        X_train = normalize(X_train, _min, _max)
        X_val = normalize(X_val, _min, _max)
        X_test = normalize(X_test, _min, _max)

        # Plot frames for manual check on the file fold10/100795-3-0-0.wav (can be removed)
        if debug:
            for i in range(T):
                plot_spec(X_train[5, i, 0], sr)

        save_filename = f"audio_data_{mnemonic}"
        if save:
            if not filename.split(".")[-1] in ["hdf5", "h5"]:
                # Save the dataset
                with open(path + save_filename, "wb") as f:
                    # protocol=4 saves correctly variables of size more than 4 GB
                    pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f, protocol=4)

    return X_train, X_val, X_test, y_train, y_val, y_test


def overlapping_split(spec: np.ndarray, num_frames: int, frame_length: int) -> np.ndarray:
    step_length = spec.shape[1] - frame_length

    '''
    window_length = s_shape[1]
    step_length = spec.shape[1] - s_shape[0]
    shape = (num_frames, window_length) + spec.shape[1:]
    strides = (spec.strides[1] * step_length,) + spec.strides
    return np.lib.stride_tricks.as_strided(spec, shape=shape, strides=strides)
    '''

    return np.array([spec[:, i: i + frame_length]
                     for i in range(0, spec.shape[1], step_length // (num_frames - 1))][:num_frames])


def contiguous_split(spec: np.ndarray, num_frames: int, frame_length: int) -> np.ndarray:
    '''
    frames = np.array_split(spec, num_frames, axis=1)
    for i in range(len(frames)):
        if frames[i].shape[1] != spec.shape[1] // num_frames:
            frames[i] = frames[i][:, :-2]
    '''
    return np.array([spec[::, i: i + frame_length] for i in
                     range(0, spec.shape[1], frame_length)][:num_frames])


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
        frames = contiguous_split(spec, num_frames, x_size)

    # Check shape and add a new axis in position 0 (this will be the channel axis)
    for i in range(frames.shape[0]):
        assert frames[i].shape == (y_size, x_size), \
            "{} should be ({},{}); instead is ({},{})".format(i, y_size, x_size, frames[i].shape[0], frames[i].shape[1])

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
    :return: the spectrogram related to the audio data contained in *data*, np.ndarray (y_size, 4 seconds)
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
    librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=sr)

    plt.colorbar()
    plt.show()


if __name__ == '__main__':

    # GENERATE DATASET

    data_path = '/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention-mechanism/'
    #postfix = 'vggish_native_4_nooverlap.pkl'

    # params
    cnn_type = 'resnet'
    #wav_path = '/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention-mechanism/UrbanSound8K/audio/fold1/7061-6-0-0.wav'
    overlap = True
    T = 10
    use_librosa = True

    p = load_hdf5(
        load_saved=False,
        save=True,
        save_filename='audio_data2.h5',
        path=data_path,
        cnn_type=cnn_type,
        overlap=overlap,
        use_librosa=use_librosa,
        debug=False
    )

    '''

    # TRY WITH SINGLE WAV FILE


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
    '''

    '''
    wav = '/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention-mechanism/UrbanSound8K/audio/fold1/7061-6-0-0.wav'
    spec1 = create_spec(
        wav,
        "resnet", 22050, 88200, 224, 224, True, overlap=True)
    spec2h = create_spec(
        wav,
        "vggish", 16000, 64000, 96, 64, False, overlap=True)
    spec2l = create_spec(
        wav,
        "vggish", 16000, 64000, 96, 64, True, overlap=True
    )

    min1 = np.min(spec1);
    min2 = np.min(spec2h);
    min3 = np.min(spec2l);
    min = np.min([min1, min2, min3])
    max1 = np.max(spec1);
    max2 = np.max(spec2h);
    max3 = np.max(spec2l);
    max = np.max([max1, max2, max3])
    spec1 = normalize(spec1, min, max)
    spec2l = normalize(spec2l, min, max)
    spec2h = normalize(spec2h, min, max)

    plot_spec(spec1, 22050)
    plot_spec(spec2l, 16000)
    plot_spec(spec2h, 16000)

    spec1 = split(spec1, 10, 224, 224, True)
    spec2h = split(spec2h, 10, 96, 64, True)
    spec2l = split(spec2l, 10, 96, 64, True)

    # specn = wavfile_to_examples(wav, return_tensor=False)[0]
    # specn=normalize(specn, min, max)
    # plot_spec(specn, 16000)

    # spec1_flat = np.swapaxes(spec1, 0, 1).reshape((224, -1))
    # spec1_flat = np.swapaxes(spec1, 0, 1).reshape((224, -1))
    # spec1_flat = np.swapaxes(spec1, 0, 1).reshape((224, -1))
    # plot_spec(np.spec1_flat, 22050)
    # plot_spec(np.concatenate(spec2l[0]), 16000)
    # plot_spec(np.concatenate(spec2h), 16000)

    for frame in spec2h:
        plot_spec(frame, 22050)
    '''