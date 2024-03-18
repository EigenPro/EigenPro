"""Optimizer class and utility functions for EigenPro iteration."""
import torch
import torch.utils.data as torch_data

import eigenpro.kernel_machine as km
import eigenpro.preconditioner as pcd
import eigenpro.data.array_dataset as array_dataset
import eigenpro.iterator as iterator
import eigenpro.projector as projector

from tqdm import tqdm


class EigenProSolver:

    def __init__(self,
                 model: km.KernelMachine,
                 data_preconditioner: pcd.Preconditioner,
                 model_preconditioner: pcd.Preconditioner,
                 tmp_centers_coeff : int = 2,
                 epochs_per_projection: int = 1,
                 ) -> None:
        """Initialize the EigenPro optimizer."""

        self.epochs_per_projection = epochs_per_projection

        self.iterator = iterator.EigenProIterator(
                model = model, 
                preconditioner = data_preconditioner,
                state_max_size = int((tmp_centers_coeff-1)*model.size),
            )
        
        self.projector = projector.EigenProProjector(
                model = model,
                preconditioner = model_preconditioner,
            )


    def fit(self, model, train_dataloader, epochs):
        
        model.train()

        for epoch in range(epochs):
            epoch_progress = tqdm(enumerate(train_dataloader), 
                total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}/{epochs}")

            for t, (x_batch, y_batch, _) in epoch_progress:

                self.iterator.step(model, x_batch, y_batch)

                # run projection when more temporary centers cannot be added or at the end of last epoch
                if ( 
                    ( # used_capacity \in (temp_model_size - batch_size, temp_model_size]
                        ## CHECK IF the `any` logic is sound for multi-device behavior
                        any(self.iterator.latent_model.used_capacity <= model.device_manager.chunk_sizes(self.iterator.state_max_size))
                        and any(self.iterator.latent_model.used_capacity > model.device_manager.chunk_sizes(self.iterator.state_max_size) - len(y_batch))
                    ) 
                    or 
                    ( # last batch of last epoch
                        (t==len(train_dataloader)-1) and (epoch==epochs-1)
                    ) 
                   ):

                    self.projector.loader.dataset.data_y = model.device_manager.gather(self.iterator.grad_accumulation)
                    
                    self.iterator.release_memory_for_projection()

                    # projection
                    for _ in range(self.epochs_per_projection):
                        for _, grad_batch, batch_ids in tqdm(
                                self.projector.loader, 
                                total=len(self.projector.loader), 
                                leave=False,
                                desc="Projection"
                            ):
                            self.projector.step(model, grad_batch, batch_ids)

                    self.iterator.reset()

        model.eval()