import time
import copy
from torch.utils.data import DataLoader
import dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import model
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 3

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Learning rate
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    test_dataloader = dataloaders.pop('test', None)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    test_acc = test_model(model, test_dataloader, criterion, optimizer)

    return model, val_acc_history, test_acc


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

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

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

    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format("test", test_loss, test_acc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return test_acc


if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load(path=sys.argv[1])
    dataloaders_dict = {
        "train": DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True),
        "val": DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False),
        "test": DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)
    }

    input_conf = "repeat"

    model_conf = [2, 2]

    cnn_conf = {
        "num_classes": 128,
        "use_pretrained": True,
        "just_bottleneck": True,
        "cnn_trainable": False,
        "first_cnn_layer_trainable": False,
        "in_channels": 3
    }

    model_ft = model.Ensemble(input_conf, cnn_conf, model_conf)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=lr)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist, test_acc = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    cnn_conf_scratch = {
        "num_classes": 128,
        "use_pretrained": False,
        "just_bottleneck": True,
        "cnn_trainable": False,
        "first_cnn_layer_trainable": False,
        "in_channels": 3
    }

    # Initialize the non-pretrained version of the model used for this run
    scratch_model = model.Ensemble(input_conf, cnn_conf_scratch, model_conf)
    scratch_model = scratch_model.to(device)

    scratch_optimizer = optim.Adam(scratch_model.parameters(), lr=lr)
    scratch_criterion = nn.CrossEntropyLoss()
    _, scratch_hist, _ = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer,
                                     num_epochs=num_epochs)

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
    plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    plt.plot(range(1, num_epochs + 1), shist, label="Scratch")
    plt.axhline(y=test_acc, linestyle='-', label="Test Accuracy")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()
