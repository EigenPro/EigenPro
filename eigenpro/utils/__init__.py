""" Common utils for package EigenPro_v3.2
"""
import torch
from .device import Device
from collections import OrderedDict

DEFAULT_DTYPE = torch.float32



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
