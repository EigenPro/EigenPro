from concurrent.futures import ThreadPoolExecutor


class MapReduceEngine:
    def __init__(self, device):
        self.device = device
        self.n_devices = len(device.devices)
        self.base_device = device.devices[0]

    def map(self, f, args_done, args_dup=None):
        # duplicate args_dup
        if args_dup != None:
            args_dup_list = self.device(args_dup)

        with ThreadPoolExecutor() as executor:
            out = [
                executor.submit(f, args_done[i], args_dup_list[i])
                for i in range(self.n_devices)
            ]

        outs = [k.result() for k in out]
        return outs

    def reduce(self, outs):
        return [out.to(self.base_device) for out in outs]
