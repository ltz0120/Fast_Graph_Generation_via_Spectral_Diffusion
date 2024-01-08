import yaml
from easydict import EasyDict as edict

import sys
import os
import os.path as osp

# sys.path.append(os.getcwd())
# ROOT_PATH = osp.abspath(osp.join(os.getcwd(), '..'))
# sys.path.append(ROOT_PATH)
print("sys.path:",sys.path)
print("os.getcwdb():",os.getcwdb())


def get_config(config, seed):
    config_dir = f'./config/{config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config