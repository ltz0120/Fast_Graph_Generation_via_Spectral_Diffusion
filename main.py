import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer
from sampler import Sampler
import pynvml



def main(work_type_args):
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    ts = time.strftime('%b%d-%H-%M-%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)
    config.type = args.beta_type

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config)
        ckpt = trainer.train(ts)
        if 'sample' in config.keys():
            config.ckpt = ckpt

            sampler = Sampler(config)
            sampler.sample()

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        print("in sample mode")
        print("config.data.data:",config.data.data)

        if config.data.data == "community_small":
            config.ckpt = ""
        elif config.data.data == "grid":
            config.ckpt = ""
        elif config.data.data == "ENZYMES":
            config.ckpt = ""

            sampler = Sampler(config) 
        sampler.sample()

    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, default="train")
    main(work_type_parser.parse_known_args()[0])
