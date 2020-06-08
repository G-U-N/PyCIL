import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data = data + class_data
            targets.append(class_targets)

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            data = data + appendent_data if len(data) != 0 else appendent_data
            targets.append(appendent_targets)

        targets = np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf)
        else:
            return DummyDataset(data, targets, trsf)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        ret_x = [x[i] for i in idxes]
        ret_y = y[idxes]
        return ret_x, ret_y


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(Image.fromarray(np.array(self.images[idx]).astype(np.uint8)))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == 'cifar10':
        return iCIFAR10()
    elif name == 'cifar100':
        return iCIFAR100()
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))
