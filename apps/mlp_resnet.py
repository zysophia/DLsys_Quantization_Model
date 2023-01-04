import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
        ), nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
         modules += ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob),
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    
    
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    terror = 0
    tloss = 0
    tsize = 0
    for i, batch in enumerate(dataloader):
        batch_x, batch_y = batch[0], batch[1]
        batch_x = batch_x.reshape((-1, 784))
        tsize += batch_x.shape[0]
        if model.training:
            opt.reset_grad()
            h = model(batch_x)
            loss = nn.SoftmaxLoss()(h, batch_y)
            training_loss = loss.numpy()
            loss.backward()
            opt.step()
        else:
            h = model(batch_x)
            loss = nn.SoftmaxLoss()(h, batch_y)
            training_loss = loss.numpy()
        y_pred = np.argmax(h.realize_cached_data(), axis=1)
        y_loss = loss.realize_cached_data()
        y_true = batch_y.realize_cached_data()
        terror += sum(y_pred != y_true)
        tloss += y_loss
    return np.array([terror/tsize, tloss/(i+1)])
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_im_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_lb_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_im_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_lb_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    train_dataset = ndl.data.MNISTDataset(train_im_path, train_lb_path)
    test_dataset = ndl.data.MNISTDataset(test_im_path, test_lb_path)
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset)
    
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in range(epochs):
        train_acc, train_loss = epoch(train_dataloader, model, opt)
    test_acc, test_loss = epoch(test_dataloader, model)
    
    return (train_acc, train_loss, test_acc, test_loss)
    
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
