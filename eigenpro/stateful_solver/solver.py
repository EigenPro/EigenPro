"""Optimizer class and utility functions for EigenPro iteration."""
import torch
import torch.utils.data as torch_data

import eigenpro.models.kernel_machine as km
import eigenpro.models.stateful_preallocated_kernel_machine as pkm
import eigenpro.preconditioners as pcd
import eigenpro.data.array_dataset as array_dataset
import eigenpro.stateful_solver.base as base 
import eigenpro.stateful_solver.iterator as iterator
import eigenpro.stateful_solver.projector as projector

from tqdm import tqdm


class EigenProSolver(base.BaseSolver):

    def __init__(self,
                 model: km.KernelMachine,
                 data_preconditioner: pcd.Preconditioner,
                 model_preconditioner: pcd.Preconditioner,
                 dtype=torch.float32,
                 tmp_centers_coeff : int = 2,
                 epochs_per_projection: int = 1,
                 ) -> None:
        """Initialize the EigenPro optimizer."""

        super().__init__(model, dtype)        
        self.epochs_per_projection = epochs_per_projection

        self.iterator = iterator.EigenProIterator(
            model = self.model, 
            preconditioner = data_preconditioner,
            temporary_model_size = int((tmp_centers_coeff-1)*model.size),
            dtype = self.dtype,)
        
        self.projector = projector.EigenProProjector(
            model = self.model,
            preconditioner = model_preconditioner,
            dtype = self.dtype,)

        # self.iterator.reset_gradient()


    def fit(self, train_dataloader, epochs):
        
        self.model.train()

        for epoch in range(epochs):
            epoch_progress = tqdm(enumerate(train_dataloader), 
                total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}/{epochs}")

            for t, (x_batch, y_batch, _) in epoch_progress:

                self.iterator.step(x_batch, y_batch)

                # run projection when more temporary centers cannot be added or at the end of last epoch
                if ( 
                    ( # used_capacity \in (temp_model_size - batch_size, temp_model_size]
                        (self.iterator.temporary_model.used_capacity <= self.iterator.temporary_model_size)
                        and (self.iterator.temporary_model.used_capacity > self.iterator.temporary_model_size - len(y_batch))
                    ) 
                    or 
                    ( # last batch of last epoch
                        (t==len(train_dataloader)-1) and (epoch==epochs-1)
                    ) 
                   ):

                    self.projector.loader.dataset.data_y = self.iterator.grad_accumulation
                    self.iterator.reset()

                    # projection
                    for _ in range(self.epochs_per_projection):
                        for _, grad_batch, batch_ids in tqdm(
                                self.projector.loader, 
                                total=len(self.projector.loader), 
                                leave=False,
                                desc="Projection"
                            ):
                            self.projector.step(grad_batch, batch_ids)

                    # self.model.eval()
                    # self.iterator.reset_gradient()
                    # self.model.train()

