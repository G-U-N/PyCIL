import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

# CIFAR100, ResNet32
# Have not found suitable hyperparameters to reproduce the paper's result yet
epochs = 90
lrate = 0.01
milestones = [50, 70]
lrate_decay = 0.1
batch_size = 128
memory_size = 0
distill_ratio = 2
attention_ratio = 0.1
weight_decay = 1e-5
num_workers = 4


class LwM(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], pretrained=False, gradcam=True)

    def after_task(self):
        self._network.zero_grad()
        self._network.unset_gradcam_hook()
        self._old_network = self._network.copy().eval()
        self._network.set_gradcam_hook()
        self._old_network.set_gradcam_hook()

        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
        # optimizer = optim.Adam(self._network.parameters(), lr=lrate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._run(train_loader, test_loader, optimizer, scheduler)

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        for epoch in range(1, epochs+1):
            self._network.train()
            clf_losses = 0.  # cross entropy
            distill_losses = 0.  # distillation
            attention_losses = 0.  # attention distillation
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs['logits']
                optimizer.zero_grad()  # Same effect as nn.Module.zero_grad()
                if self._old_network is None:
                    clf_loss = F.cross_entropy(logits, targets)
                    clf_losses += clf_loss.item()
                    loss = clf_loss
                else:
                    self._old_network.zero_grad()
                    old_outputs = self._old_network(inputs)
                    old_logits = old_outputs['logits']

                    # Classification loss
                    # if no old samples saved, only calculate loss for new logits
                    clf_loss = F.cross_entropy(logits[:, self._known_classes:], targets - self._known_classes)
                    clf_losses += clf_loss.item()

                    # Distillation loss
                    # if no old samples saved, only calculate distillation loss for old logits
                    '''
                    distill_loss = F.binary_cross_entropy_with_logits(
                        logits[:, :self._known_classes], torch.sigmoid(old_logits.detach())
                    ) * distill_ratio
                    '''
                    distill_loss = _KD_loss(logits[:, :self._known_classes], old_logits.detach(), T=2) * distill_ratio
                    distill_losses += distill_loss.item()

                    # Attention distillation loss
                    top_base_indices = logits[:, :self._known_classes].argmax(dim=1)
                    onehot_top_base = target2onehot(top_base_indices, self._known_classes).to(self._device)

                    logits[:, :self._known_classes].backward(gradient=onehot_top_base, retain_graph=True)
                    old_logits.backward(gradient=onehot_top_base)

                    attention_loss = gradcam_distillation(
                        outputs['gradcam_gradients'][0], old_outputs['gradcam_gradients'][0].detach(),
                        outputs['gradcam_activations'][0], old_outputs['gradcam_activations'][0].detach()
                    ) * attention_ratio
                    attention_losses += attention_loss.item()

                    # Integration
                    loss = clf_loss + distill_loss + attention_loss

                    self._old_network.zero_grad()
                    self._network.zero_grad()

                optimizer.zero_grad()  # Same effect as nn.Module.zero_grad()
                loss.backward()
                optimizer.step()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            # train_acc = self._compute_accuracy(self._network, train_loader)
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info1 = 'Task {}, Epoch {}/{} => clf_loss {:.2f}, '.format(self._cur_task, epoch, epochs, clf_losses/(i+1))
            info2 = 'distill_loss {:.2f}, attention_loss {:.2f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                distill_losses/(i+1), attention_losses/(i+1), train_acc, test_acc)
            logging.info(info1 + info2)


def gradcam_distillation(gradients_a, gradients_b, activations_a, activations_b):
    attentions_a = _compute_gradcam_attention(gradients_a, activations_a)
    attentions_b = _compute_gradcam_attention(gradients_b, activations_b)

    assert len(attentions_a.shape) == len(attentions_b.shape) == 4
    assert attentions_a.shape == attentions_b.shape

    batch_size = attentions_a.shape[0]

    flat_attention_a = F.normalize(attentions_a.view(batch_size, -1), p=2, dim=-1)
    flat_attention_b = F.normalize(attentions_b.view(batch_size, -1), p=2, dim=-1)

    distances = torch.abs(flat_attention_a - flat_attention_b).sum(-1)

    return torch.mean(distances)


def _compute_gradcam_attention(gradients, activations):
    alpha = F.adaptive_avg_pool2d(gradients, (1, 1))
    return F.relu(alpha * activations)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / soft.shape[0]
