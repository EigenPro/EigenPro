"""Optimizer class and utility functions for EigenPro iteration."""
import torch
from .models.base import KernelMachine
from .preconditioner import Preconditioner
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class EigenPro:
    """EigenPro optimizer for kernel machines.

    Args:
        model (KernelMachine): A KernelMachine instance.
        threshold_index (int): An index used for thresholding.
        data_preconditioner (Preconditioner): Preconditioner instance that contains a
            top kernel eigensystem for correcting the gradient for data.
        model_preconditioner (Preconditioner): Preconditioner instance that contains a
            top kernel eigensystem for correcting the gradient for the projection

    Attributes:
        model (KernelMachine): A KernelMachine instance.
        precon (Preconditioner): A Preconditioner instance.
        _threshold_index (int): An index used for thresholding.
    """

    def __init__(self,
                 model: KernelMachine,
                 threshold_index: int,
                 data_preconditioner: Preconditioner,
                 model_preconditioner: Preconditioner,
                 kz_xs_evecs:torch.tensor = None,
                 dtype=torch.float32,
                 accumulated_gradients:bool = False,) -> None:
        """Initialize the EigenPro optimizer."""

        self.dtype = dtype
        self._model = model
        self._threshold_index = threshold_index
        self.data_preconditioner  = data_preconditioner
        self.model_preconditioner = model_preconditioner

        if accumulated_gradients:
            self.grad_accumulation = [torch.tensor(0,dtype=self.dtype).to(self.model.device.devices[i])
                                      for i in range(len(self.model.device.devices)) ]
            if kz_xs_evecs == None:
                raise NotImplementedError
            else:
                self.k_centers_nystroms_mult_eigenvecs = kz_xs_evecs
        else:
            self.grad_accumulation = None

        #### adding nystrom samples to the model
        self._model.add_centers(data_preconditioner.centers.to(dtype), None,nystrom_centers = True)



    @property
    def model(self) -> KernelMachine:
        """Gets the active model (for training).

        Returns:
            KernelMachine: The active model.
        """
        return self._model


    def step(self,
             batch_x: torch.Tensor,
             batch_y: torch.Tensor,
             batch_ids: torch.Tensor,
             gpu_ind = 0,
             projection:bool=False) -> None:
        """Performs a single optimization step.

        Args:
            batch_x (torch.Tensor): Batch of input features.
            batch_y (torch.Tensor): Batch of target values.
            batch_ids (torch.Tensor): Batch of sample indices.
            projection (bool): projection mode
        """

        # t_forward_start = time.time()
        batch_p = self.model.forward(batch_x,projection=projection)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # print(f'forward time:{time.time()-t_forward_start}')

        base_device = batch_p.device
        grad = batch_p - batch_y.to(self.dtype).to(base_device) ## gradient in function space K(bathc,.) (f-y)
        batch_size = batch_x.shape[0]


        if projection:
            lr = self.model_preconditioner.scaled_learning_rate(batch_size)
            deltap, delta = self.model_preconditioner.delta(batch_x.to(grad.device).to(self.dtype), grad)
        else:
            lr = self.data_preconditioner.scaled_learning_rate(batch_size)
            deltap, delta = self.data_preconditioner.delta(batch_x.to(grad.device).to(self.dtype), grad.to(self.dtype))

        if self.grad_accumulation is None:# or projection:
            self.model.update_by_index(batch_ids, -lr *grad)#,projection=projection )

        elif projection:
            self.model.update_projection(batch_ids, -lr *grad, gpu_index=gpu_ind)

        else:
            k_centers_batch_all = [m.lru.get('k_centers_batch') for m in self.model.shard_kms ]
            self.model.lru.cache.clear()

            grads = self.model.device(grad, strategy="broadcast")  # of shape (batch_size,)
            deltaps = self.model.device(deltap, strategy="broadcast")  # of shape (nystrom_size,)
            with ThreadPoolExecutor() as executor:
                kgrads = [
                    executor.submit(
                        torch.matmul,
                        k_centers_batch_all[i],  # of shape (model_size/num_gpus, batch_size)
                        grads[i],  # of shape (batch_size,)
                    ) for i in range(len(self.model.device.devices))
                ]
                delta_grad = [
                    executor.submit(
                        torch.matmul,
                        self.k_centers_nystroms_mult_eigenvecs[i],  # of shape (model_size/num_gpus, nystrom_size)
                        deltaps[i],  # of shape (nystrom_size)
                    ) for i in range(len(self.model.device.devices))
                ]
                # import ipdb
                # ipdb.set_trace()
                self.grad_accumulation = [
                    executor.submit(
                        torch.add,
                        self.grad_accumulation[i],  # of shape (model_size/num_gpus,)
                        -lr.item() * kgrads[i].result()
                        + lr.item() * delta_grad[i].result(),  # of shape (model_size/num_gpus)
                    ).result() for i in range(len(self.model.device.devices))
                ]

            # kgrads = []
            # for k in k_centers_batch_all:
            #     kgrads.append((k @ grad.to(k.device).to(k.dtype)).to(base_device))
            # k_centers_batch_grad = torch.cat(kgrads)  ##  K(bathc,Z) (f-y)

            # self.grad_accumulation = self.grad_accumulation - lr*\
            #                          ( k_centers_batch_grad -
            #                            (self.k_centers_nystroms_mult_eigenvecs @
            #                             deltap) )

            self.model.add_centers(batch_x, -lr*grad)
            # print(f'used capacity:{self.model.shard_kms[0].used_capacity}')
            del  kgrads, k_centers_batch_all,batch_y





        self.model.update_by_index(None,lr*delta, nystrom_update=True,projection=projection)

        del grad, batch_x, batch_p, deltap, delta
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def reset(self):
        """Reset the gradient accumulation
        Args:
            None
        return:
            None
        """
        self.grad_accumulation = [torch.tensor(0,dtype=self.dtype).to(self.model.device.devices[i])
                                      for i in range(len(self.model.device.devices)) ]

