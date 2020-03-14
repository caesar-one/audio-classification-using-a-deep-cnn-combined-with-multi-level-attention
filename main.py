from dataset import load
from model import Ensemble
import torch
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm

N_EPOCHS = 50

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load(save=False, load_saved=True)
    '''
    train_data = data.TensorDataset(X_train, y_train)

    # print some info
    print(train_data[0])
    print(len(train_data))

    train_data_loader = data.DataLoader(train_data, batch_size=512, shuffle=True)

    use_cuda = torch.cuda.is_available()
    print("CUDA available: " + str(use_cuda))
    device = torch.device("cuda" if use_cuda else "cpu")

    net = Ensemble([2, 2, 2])
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(params=net.parameters(), lr=0.001)  # learning rate: pace of weights update

    net.train()
    for epoch in tqdm('Training...', range(N_EPOCHS)):
        for Xb, yb in train_data_loader:  # extracts batches from the dataset
            y_pred = net(Xb)
            loss_epoch = loss(y_pred, yb)
            loss_epoch.backward()
            opt.step()
            opt.zero_grad()
            '''
