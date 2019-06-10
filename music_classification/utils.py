from typing import Mapping


def train_epoch():
    # TODO this function realization depends on train_model only
    pass


def train_model(model,
                data: Mapping[str, tuple],
                verbose: bool = True,
                batch_size: int = 32,
                random_state: int = 13,
                dump_iter: int = 5,
                save_dir: str = './res') -> dict:
    """
    Train model
    

    :param model: CRNN model
    :param data: dict with keys 'train', 'validation', 'test' and values (X, Y, names)
        X - np.array(shape=[samples, mel_len, slice_length)
        Y - np.array(shape=[samples,]) - str
        names - names of songs
    :param verbose: verbose
    :param batch_size: size of batch
    :param random_state: random state for torch, numpy
    :param dump_iter: number of iters between dump of models weights
    :param save_dir: path to save model weights with format model_{num_iter}.model and model_final.model
    :return:

    dict with keys:
    'model' - trained_model
    'val_score', 'train_score', 'test_score' - validation, train and test scores, float
    'train_losses', 'val_losses', 'test_losses' - losses on each epoch, list
    """
    # TODO add other arguments(reccomended to add only with default values)
    # TODO choose tensorboard(erase losses from return) or not
    # TODO add arguments to
    pass
