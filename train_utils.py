import os.path

import torch

from utils import make_directory_if_no_exists


def save_checkpoint(model, optimizer, train_loss_list, val_loss_list, model_fname, directory):
    """
    Save model checkpoints.
    :param model:
    :param optimizer:
    :param train_loss_list:
    :param val_loss_list:
    :param model_fname:
    :param directory:
    :return:
    """
    # ensure fname ends with .pt
    if not model_fname.endswith('.pt'):
        model_fname = f'{model_fname}.pt'

    # save checkpoint
    make_directory_if_no_exists(directory)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
    }, os.path.join(directory, model_fname))
    return


def load_checkpoint(model, optimizer, model_fname, directory):
    """
    Load model checkpoints.
    :param model:
    :param optimizer:
    :param model_fname:
    :param directory:
    :return:
    """
    # ensure fname ends with .pt
    if not model_fname.endswith('.pt'):
        model_fname = f'{model_fname}.pt'

    # load checkpoint
    checkpoint = torch.load(os.path.join(directory, model_fname))
    # init model with loaded state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    # init optimizer with loaded state dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # get loaded loss values
    train_loss_list = checkpoint['train_loss_list']
    val_loss_list = checkpoint['val_loss_list']
    return (model, optimizer, train_loss_list, val_loss_list)


def save_metrics(train_loss_list, valid_loss_list, global_steps_list, directory):
    """
    Load metrics.
    :param train_loss_list:
    :param valid_loss_list:
    :param global_steps_list:
    :param directory:
    :return:
    """

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, directory)
    return


def load_metrics(directory, device):
    """
    Save metrics.
    :param directory:
    :return:
    """
    state_dict = torch.load(directory, map_location=device)
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']