'''
We implemented `iCaRL+RMM`, `FOSTER+RMM` in [rmm.py](models/rmm.py).  We implemented the `Pretraining Stage` of `RMM` in [rmm_train.py](rmm_train.py). 
Use the following training script to run it.
```bash
python rmm_train.py --config=./exps/rmm-pretrain.json
```
'''
import json
import argparse
from trainer import train
import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.rl_utils.ddpg import DDPG
from utils.rl_utils.rl_utils import ReplayBuffer
from utils.toolkit import count_parameters
import os
import numpy as np
import random


class CILEnv:
    def __init__(self, args) -> None:
        self._args = copy.deepcopy(args)
        self.settings = [(50, 2), (50, 5), (50, 10), (50, 20), (10, 10), (20, 20), (5, 5)]
        # self.settings = [(5,5)] #  Debug
        self._args["init_cls"], self._args["increment"] = self.settings[np.random.randint(len(self.settings))]
        self.data_manager = DataManager(
            self._args["dataset"],
            self._args["shuffle"],
            self._args["seed"],
            self._args["init_cls"],
            self._args["increment"],
        )
        self.model = factory.get_model(self._args["model_name"], self._args)

    @property
    def nb_task(self):
        return self.data_manager.nb_tasks

    @property
    def cur_task(self):
        return self.model._cur_task

    def get_task_size(self, task_id):
        return self.data_manager.get_task_size(task_id)

    def reset(self):
        self._args["init_cls"], self._args["increment"] = self.settings[np.random.randint(len(self.settings))]
        self.data_manager = DataManager(
            self._args["dataset"],
            self._args["shuffle"],
            self._args["seed"],
            self._args["init_cls"],
            self._args["increment"],
        )
        self.model = factory.get_model(self._args["model_name"], self._args)

        info = "start new task:  dataset: {}, init_cls: {},  increment: {}".format(
            self._args["dataset"], self._args["init_cls"], self._args["increment"]
        )
        return np.array([self.get_task_size(0) / 100, 0]), None, False, info

    def step(self, action):
        self.model._m_rate_list.append(action[0])
        self.model._c_rate_list.append(action[1])
        self.model.incremental_train(self.data_manager)
        cnn_accy, nme_accy = self.model.eval_task()
        self.model.after_task()
        done = self.cur_task == self.nb_task - 1
        info = "running task [{}/{}]:  dataset: {}, increment: {}, cnn_accy top1: {},  top5: {}".format(
            self.model._known_classes,
            100,
            self._args["dataset"],
            self._args["increment"],
            cnn_accy["top1"],
            cnn_accy["top5"],
        )
        return (
            np.array(
                [
                    self.get_task_size(self.cur_task+1)/100 if not done else 0.,
                    self.model.memory_size
                    / (self.model.memory_size + self.model.new_memory_size),
                ]
            ),
            cnn_accy["top1"]/100,
            done,
            info,
        )


def _train(args):

    logs_name = "logs/RL-CIL/{}/".format(args["model_name"])
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/RL-CIL/{}/{}_{}_{}_{}_{}".format(
        args["model_name"],
        args["prefix"],
        args["seed"],
        args["model_name"],
        args["convnet_type"],
        args["dataset"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)

    actor_lr = 5e-4
    critic_lr = 5e-3
    num_episodes = 200
    hidden_dim = 32
    gamma = 0.98
    tau = 0.005
    buffer_size = 1000
    minimal_size = 50
    batch_size = 32
    sigma = 0.2 # action noise, encouraging the off-policy algo to explore.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = CILEnv(args)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DDPG(
        2, 1, 4, hidden_dim, False, 1, sigma, actor_lr, critic_lr, tau, gamma, device
    )
    for iteration in range(num_episodes):
        state, *_, info = env.reset()
        logging.info(info)
        done = False
        while not done:
            action = agent.take_action(state)
            logging.info(f"take action: m_rate {action[0]}, c_rate {action[1]}")
            next_state, reward, done, info = env.step(action)
            logging.info(info)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    "states": b_s,
                    "actions": b_a,
                    "next_states": b_ns,
                    "rewards": b_r,
                    "dones": b_d,
                }
                agent.update(transition_dict)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple continual learning algorthms."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/finetune.json",
        help="Json file of settings.",
    )

    return parser


if __name__ == "__main__":
    main()
