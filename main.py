from utils import create_kernelmodel

X,Y,Z  = load(dataset)
loader = Loader(X,y)

S_data = sample(X)
S_centers = sample(Z)

model = create_KernelModels(Z,n_outputs,kernel_fn) 
precon_data = Preconditioner(S_data,top_q_data)
precon_model = Preconditioner(S_centers,top_q_centers)
projection = Projector(Z)
T = 10
optim = Eigenpro(model,precon)

for t,x_batch,y_batch in enumerate(loader):
    optim.step(x_batch,y_batch)
    if (t+1)%T==0:
        projector(optim.model)
projector(optim.model)

