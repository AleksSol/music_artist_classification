from typing import Mapping

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, TensorDataset
from torchtools.trainer import Trainer
from torchtools.meters import LossMeter, AccuracyMeter
from torchtools.callbacks import TensorBoardLogger, CSVLogger, ModelCheckPoint, EarlyStopping
from torchtools.exceptions import EarlyStoppingException


class EarlyStoppingNoRaise(EarlyStopping):
    def on_epoch_end(self, trainer, state):
        try:
            super(EarlyStoppingNoRaise, self).on_epoch_end(trainer, state)
        except EarlyStoppingException:
            trainer.exit()


def train_epoch():
    # TODO this function realization depends on train_model only
    pass


def train_model(model,
                data: Mapping[str, tuple],
                batch_size: int = 32,
                random_state: int = 13,
                save_dir: str = './res',
                n_epochs: int = 100,
                learning_rate: float = 1e-4,
                device: str='cuda') -> dict:
    """
    Train model
    

    :param model: CRNN model
    :param data: dict with keys 'train', 'validation', 'test' and values (X, Y, names)
        X - np.array(shape=[samples, mel_len, slice_length)
        Y - np.array(shape=[samples,]) - str
        names - names of songs
    :param batch_size: size of batch
    :param random_state: random state for torch, numpy
    :param save_dir: path to save model weights with format model_{num_iter}.model and model_final.model
    :param n_epochs: number of epochs to train for
    :param learning_rate: learning rate for Adam
    :param device: device to train on e.g. 'cpu' or 'cuda' or 'cuda:n'
    :return:

    dict with keys:
    'model' - trained_model
    'val_score', 'train_score', 'test_score' - validation, train and test scores, float
    'train_losses', 'val_losses', 'test_losses' - losses on each epoch, list
    """

    torch.manual_seed(random_state)

    if 'cuda' in device:
        torch.cuda.manual_seed(random_state)
        #cudnn.benchmark = True

    dev = torch.device(device)
    model.to(dev)
    
    def make_dataset_and_loader(data, shuffle=True):
        X, y = [torch.from_numpy(t) for t in data[:2]]
        X = X.unsqueeze(dim=1).type(torch.float32)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, loader

    train_set, train_loader = make_dataset_and_loader(data['train'])
    val_set, val_loader = make_dataset_and_loader(data['validation'], shuffle=False)
    test_set, test_loader = make_dataset_and_loader(data['test'], shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, train_loader, criterion, optimizer, val_loader, test_loader, device=dev)
    loss, val_loss = LossMeter('loss'), LossMeter('val_loss')
    acc, val_acc = AccuracyMeter('acc'), AccuracyMeter('val_acc')

    logger = TensorBoardLogger()
    csv_logger = CSVLogger(keys=['epochs', 'loss', 'acc', 'val_loss', 'val_acc'])
    checkpoint = ModelCheckPoint(save_dir, fname='model_{epochs:03d}.model', save_best_only=False)
    early_stop = EarlyStoppingNoRaise(monitor='val_loss', patience=10, mode='auto')

    hooks = [loss, val_loss, acc, val_acc, logger, csv_logger, checkpoint, early_stop]
    trainer.register_hooks(hooks)

    trainer.train(n_epochs)
    logs = pd.read_csv(csv_logger.fpath)

    return dict(
        model=model,
        train_score=np.array(logs['acc'])[-1],
        val_score=np.array(logs['val_acc'])[-1],
        train_losses=np.array(logs['loss']),
        val_losses=np.array(logs['val_loss'])
    )
