import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                         decimals=2)

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
