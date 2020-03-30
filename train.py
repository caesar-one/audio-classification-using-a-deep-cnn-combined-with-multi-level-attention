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
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import os

import matplotlib.pyplot as plt
from glob import glob

import dataset
import model
from params import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class H5Loader(Dataset):

    def __init__(self, X_desc, y_desc):
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
    if not save_model_path: save_model_path = "best_weights.h5"
    save_model(clf, os.path.splitext(save_model_path)[0] + ("_final_finetuned" if finetune else "_final") + os.path.splitext(save_model_path)[1])

    return clf, val_acc_history, test_acc


def test_model(model, dataloader, criterion, optimizer):
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

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
def trainable_params(model, feature_extract):
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

    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load()

    '''
    dataloaders_dict = {
        "train": DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(list(zip(X_val, y_val)), batch_size=BATCH_SIZE, shuffle=False),
        "test": DataLoader(list(zip(X_test, y_test)), batch_size=BATCH_SIZE, shuffle=False)
    }
    '''

    dataloaders_dict = {
        "train": DataLoader(H5Loader(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        "val": DataLoader(H5Loader(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
        "test": DataLoader(H5Loader(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    input_conf = "repeat"

    model_conf = [2, 2]

    cnn_conf = {
        "cnn_type": "vggish",
        "num_classes": 10,
        "use_pretrained": True,
        "just_bottleneck":False,
        "cnn_trainable": False,
        "first_cnn_layer_trainable": False,
        "in_channels": 3
    }

    model_ft = model.Ensemble(input_conf, cnn_conf, model_conf, device)

    # Send the model to GPU
    #model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(trainable_params(model_ft, feature_extract=FEATURE_EXTRACT), lr=LR)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist, test_acc = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)

    cnn_conf_scratch = {
        "cnn_type": "vggish",
        "num_classes": 10,
        "use_pretrained": False,
        "just_bottleneck":False,
        "cnn_trainable": False,
        "first_cnn_layer_trainable": False,
        "in_channels": 3
    }

    # Initialize the non-pretrained version of the model used for this run
    scratch_model = model.Ensemble(input_conf, cnn_conf_scratch, model_conf, device)
    scratch_model = scratch_model.to(device)

    scratch_optimizer = optim.Adam(scratch_model.parameters(), lr=LR)
    scratch_criterion = nn.CrossEntropyLoss()
    _, scratch_hist, _ = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer,
                                     num_epochs=NUM_EPOCHS)

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    # ohist = []
    # shist = []

    ohist = [h.cpu().numpy() for h in hist]
    shist = [h.cpu().numpy() for h in scratch_hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, len(ohist) + 1), ohist, label="Pretrained")
    plt.plot(range(1, len(shist) + 1), shist, label="Scratch")
    plt.axhline(y=test_acc, linestyle='-', label="Test Accuracy")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, NUM_EPOCHS + 1, 1.0))
    plt.legend()
    plt.show()
