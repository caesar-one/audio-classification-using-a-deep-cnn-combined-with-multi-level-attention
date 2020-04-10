try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import time
import copy
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

import dataset
import model
from params import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class H5Loader(Dataset):

    def __init__(self, X_desc, y_desc):
        """
        A PyTorch iterable dataset, suitable for both HDF5 file descriptors or NumPy arrays.

        :param X_desc: HDF5 file descriptor or NumPy array for the features
        :param y_desc: HDF5 file descriptor or NumPy array for the labels
        :return: An iterator that can be used with the torch.utils.data.DataLoader class.
        """
        self.X_desc = X_desc
        self.y_desc = y_desc

    def __len__(self):
        return self.y_desc.shape[0]

    def __getitem__(self, idx):
        return (self.X_desc[idx], self.y_desc[idx])

# if patience=None the early stopping mechanism will not be active. Otherwise, if patience=N training will be stopped
#       if there will not be improvements for N epochs (on the validation set). If save_model_path=None, the model won't
#       be saved. Otherwise it will be saved in the specified path.
def train_model(clf, dataloaders, criterion, optimizer, num_epochs=25,
                patience=10, save_model_path=None, resume=False, finetune=False):
    """
    Trains a PyTorch *model* for *num_epochs* epochs using *criterion* loss function and *optimizer* optimizer.
        Patience, early stopping and automatic checkpointing are also possible by setting the respective vars.

    :param clf: an instantiated PyTorch model
    :param dataloaders: a dict of DataLoaders with keys corresponding to the set name, e.g. "train" or "val"
    :param criterion: a PyTorch criterion instance
    :param optimizer: a PyTorch optimizer instance
    :param num_epochs: the number of epochs to train the model on
    :param patience: can be None or integer: stops training if "val" set accuracy does not improve for *patience* epochs
    :param save_model_path: can be None or str, saves a checkpoint in the specified path at each "val" acc improvement
    :param resume: if True, loads a pretrained model from *save_model_path* path
    :param finetune: if True, makes every *clf* parameter trainable
    :return: a tuple containing the trained model with best "val" acc weights, the "val" acc history and the test acc
    """

    since = time.time()

    val_acc_history = []

    #best_model_wts = copy.deepcopy(clf.state_dict())
    best_acc = 0.0
    best_epoch = 0
    epoch = 0

    if resume:
        assert save_model_path is not None
        if save_model_path in glob(save_model_path):
            _model, _criterion, _optimizer, _epoch, _loss, _accuracy, _history = _resume_from_checkpoint(save_model_path)
            #if finetune:
            #    model.set_requires_grad(_model, True)
            clf = _model
            criterion = _criterion
            optimizer = _optimizer
            epoch = _epoch + 1
            best_epoch = _epoch
            best_acc = _accuracy
            val_acc_history = _history
        else:
            raise Exception("No such model file in the specified path.")

    if finetune:
        model.set_requires_grad(clf, True)

    best_model_wts = copy.deepcopy(clf.state_dict())
    test_dataloader = dataloaders.pop('test', None)

    clf = clf.to(device)

    for epoch in range(epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                clf.train()  # Set model to training mode
            else:
                clf.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = clf(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(clf.state_dict())
                best_epoch = epoch
                if save_model_path:
                    _save_checkpoint(clf, criterion, optimizer, epoch, epoch_loss, best_acc, val_acc_history, save_model_path)
                    print("Model checkpoint saved successfully in the given path!")
        print()
        if patience is not None:
            if epoch - best_epoch >= patience:
                break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    clf.load_state_dict(best_model_wts)

    test_acc = test_model(clf, test_dataloader, criterion, optimizer)
    if not save_model_path:
        if IN_COLAB:
            save_model_path = "/content/drive/My Drive/Audio-classification-using-multiple-attention-mechanism/best_weights.h5"
        else:
            save_model_path = "best_weights.h5"
    save_model(clf, os.path.splitext(save_model_path)[0] + ("_final_finetuned" if finetune else "_final") + os.path.splitext(save_model_path)[1])

    return clf, val_acc_history, test_acc


def test_model(model, dataloader, criterion, optimizer):
    """
    Performs a test on a PyTorch *model* using the *dataloader*

    :param model: the model to test
    :param dataloader: an instance of torch.utils.data.DataLoader
    :param criterion: the loss function
    :param optimizer: the optimizer (in reality, this is not needed)
    :return: a tuple containing the test acc and a dict containing a summary of the test results
    """
    if dataloader is None:
        return None

    since = time.time()

    # Final testing phase
    print('Testing')
    print('-' * 10)

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.f
    metric_pred, metric_true = [], []
    for inputs, labels in tqdm(dataloader,"Testing"):
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        metric_pred.append(preds)
        metric_true.append(labels.data)

    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)

    print('{} Loss: {:.4f}, Acc: {:.4f}'.format("test", test_loss, test_acc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    metric_true = torch.cat(metric_true, 0).cpu()
    metric_pred = torch.cat(metric_pred, 0).cpu()

    print(classification_report(metric_true,metric_pred,target_names=TARGET_NAMES,digits=3))

    cm = confusion_matrix(metric_true, metric_pred).astype(np.float32)
    for i in range(cm.shape[0]):
        _sum = sum(cm[i])
        for j in range(cm.shape[1]):
            cm[i, j] = cm[i, j] * 100 / _sum

    disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES, values_format='.0f'))
    disp.plot(xticks_rotation='vertical', cmap='Blues')

    results = classification_report(metric_true,metric_pred,target_names=TARGET_NAMES, output_dict=True, digits=5)
    return test_acc, results


def _save_checkpoint(model, criterion, optimizer, epoch, loss, accuracy, history, path):
    torch.save({
        'epoch': epoch,
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'loss': loss,
        'accuracy': accuracy,
        'history': history
    }, path)

def _resume_from_checkpoint(path):
    d = torch.load(path)
    return d["model"], d["criterion"], d["optimizer"], d["epoch"], d["loss"], d["accuracy"], d["history"]


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_args, path):
    m = model.Ensemble(**model_args)
    m.load_state_dict(torch.load(path, map_location=device))
    return m

def trainable_params(model, feature_extract):
    """
    Prints and returns all the trainable parameters in model.

    :param model: the model instance
    :param feature_extract: if True, only params with *requires_grad* will be returned.
    :return: a list containing the model's trainable params.
    """
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    return params_to_update


if __name__ == "__main__":
    import h5py

    def range_aux(start, end, step):
        for i in tqdm(range(start, end, step)):
            yield (i, i + step) if i + step <= end else (i, end)

    data_path = '/Volumes/GoogleDrive/Il mio Drive/Audio-classification-using-multiple-attention-mechanism/'

    overlap = True
    use_librosa = True
    cnn_type = "resnet"

    mnemonic = f'{cnn_type}{"" if use_librosa else "_native"}_{T}{"_s" if overlap else ""}'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extract = True
    # lr = 0.001

    input_conf = "repeat"

    model_conf = [2, 1]

    cnn_conf = {
        "cnn_type": cnn_type,
        "num_classes": 128,
        "use_pretrained": True,
        "just_bottlenecks": True,
        "cnn_trainable": False,
        "first_cnn_layer_trainable": False,
        "in_channels": 3}

    # save_model_path = f"{data_path}model_{mnemonic}.pkl"
    model_ft = model.Ensemble(input_conf, cnn_conf, model_conf, device)

    import gc
    import h5py

    load_in_RAM = False

    #audio_data = h5py.File("audio_data_c.h5", "r")
    audio_data = h5py.File(data_path + "audio_data_3.h5", "r")
    group = audio_data["resnet_10_s"]
    #X_train = group["X_train"]
    #X_val = group["X_val"]
    X_test = group["X_test"]
    #y_train = group["y_train"]
    #y_val = group["y_val"]
    y_test = group["y_test"]

    if load_in_RAM:
        gc.collect()
        #X_train = X_train[:]
        #X_val = X_val[:]
        X_test = X_test[:]
        #y_train = y_train[:]
        #y_val = y_val[:]
        y_test = y_test[:]

        audio_data.close()

    batch_size = 64

    optimizer = optim.Adam(trainable_params(model_ft, True), lr=0.001)
    optimizer_ft = optim.Adam(trainable_params(model_ft, True), lr=0.0001)

    criterion = nn.CrossEntropyLoss()

    print(sum(p.numel() for p in model_ft.parameters() if p.requires_grad))

    print('Loading model...')
    # Load model
    model_ft = load_model({
        'input_conf': input_conf,
        'cnn_conf': cnn_conf,
        'model_conf': model_conf,
        'device': device},
        data_path + mnemonic + '_2A1A/wts_691.h5')
    print('Model loaded.')

    #model_ft.to(device)

    # Train and evaluate
    # model_ft, hist, test_acc = train.train_model(model_ft, dataloaders_dict, criterion, optimizer, resume=False,
    #                                              num_epochs=25, finetune=False)

    #torch.cuda.empty_cache()
    batch_size = 8

    dataloaders_dict = {
        #"train": DataLoader(H5Loader(X_train, y_train), batch_size=batch_size, shuffle=True),
        #"val": DataLoader(H5Loader(X_val, y_val), batch_size=batch_size, shuffle=False),
        "test": DataLoader(H5Loader(X_test, y_test), batch_size=batch_size, shuffle=False)
    }



