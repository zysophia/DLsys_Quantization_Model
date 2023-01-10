import sys

from torch.nn import parameter
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None, iter_limit=10000):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        if opt is not None:
            opt.reset_grad()
        X, y = batch
        X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        out = model(X)
        curcorrect = np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        correct += curcorrect
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy()[0] * y.shape[0]
        if opt is not None:
            loss.backward()
            opt.step()
        if i%5 == 0:
          print("cifar iter", i, "acc", curcorrect/(y.shape[0]), 'loss', loss.data.numpy()[0])
        i += 1
        if i>iter_limit:
          break
    return correct/(y.shape[0]*i), total_loss/(y.shape[0]*i)

    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss, iter_limit=10000):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    tacc, tloss = 0.0, 0.0
    ### BEGIN YOUR SOLUTION
    opt=optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for e in range(n_epochs):
      ret = epoch_general_cifar10(dataloader, model, loss_fn(), opt, iter_limit)
      tacc += ret[0]
      tloss += ret[1]
    return np.array([tacc/n_epochs, tloss/n_epochs])

    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    ret = epoch_general_cifar10(dataloader, model, loss_fn())
    return np.array([ret[0], ret[1]])
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    # corpus = ndl.data.Corpus("./data/ptb")
    # seq_len = 40
    # batch_size = 16
    # hidden_size = 100
    # train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    # model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    # train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
    dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=False
             # collate_fn=ndl.data.collate_ndarray,
             # drop_last=False,
             # device=device,
             # dtype="float32"
             )
    from apps.models import ResNet9
    model = ResNet9(device=device, dtype="float32")
    epochs = 1
    out = train_cifar10(model, dataloader, n_epochs=epochs)
    print("-------training cifar10 with resnet9 on", epochs, "epocs------")
    print("avg acc: ", out[0], " avg loss: ", out[1])






