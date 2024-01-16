""" Common utils for package EigenPro_v3.2
"""
import torch
from .device import Device
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import argparse
import ipdb

DEFAULT_DTYPE = torch.float32

      
class MapReduceEngein():
    def __init__(self, device):
        self.device = device
        self.n_devices = len(device.devices)
        self.base_device = device.devices[0]

    def map(self,f,args_done , args_dup = None):
        # duplicate args_dup
        if args_dup != None:
            args_dup_list = self.device(args_dup)

        with ThreadPoolExecutor() as executor:
            out = [executor.submit(f,args_done[i], args_dup_list[i]) for i in range(self.n_devices)]

        outs = [k.result() for k in out]
        # # split arg in args_splits
        # for arg in args_split:
        #     splitted = split(arg, devices)
        #
        # # call f on each device
        return outs


    def reduce(self,outs):
        return [out.to(self.base_device) for out in outs]


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="Number of data points",default=50_000)
    parser.add_argument("--p", type=int, help="Model size",default=20_000)
    parser.add_argument("--s_data", type=int, help="s_data",default=2_000)
    parser.add_argument("--s_model", type=int, help="s_model",default=2_000)
    parser.add_argument("--q_data", type=int, help="q_data",default=100)
    parser.add_argument("--q_model", type=int, help="q_model",default=100)
    parser.add_argument("--epochs", type=int, help="q_model",default=2)
    return parser.parse_args()    



####################################################################
# wandb_init = {}
# wandb_init["project_name"] ="---"
# wandb_init["mode"]  = 'online'
# wandb_init["key"]  = "----"
# wandb_init["org"] = "----"
# os.environ["WANDB_API_KEY"] = wandb_init['key']
# os.environ["WANDB_MODE"] = wandb_init['mode']  # online or offline
# run = wandb.init(project=wandb_init['project_name'], \
#                  entity=wandb_init['org'])
# run.name = f'---'
# run.save()
####################################################################
