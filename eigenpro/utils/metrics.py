import torch


def get_performance(model, X, Y, batch_size=1024):
    device_manager = model.device_manager #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    # Ensure model is in evaluation mode
    # Convert X and Y to PyTorch datasets and use DataLoader for batch processing
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            # Move batch to GPU
            batch_X, batch_Y = device_manager.broadcast(batch_X.to(model.dtype)), batch_Y.to(model.dtype).to(device_manager.base_device)

            # Forward pass
            Y_hat = device_manager.reduce_add(model(batch_X))

            # Calculate loss and accuracy
            loss = torch.norm(Y_hat - batch_Y)**2 / batch_Y.size(0)
            accuracy = torch.sum(torch.argmax(Y_hat, dim=1) == torch.argmax(batch_Y, dim=1)).item()

            # Accumulate loss and accuracy
            total_loss += loss.item() * batch_Y.size(0)
            total_accuracy += accuracy
            total_samples += batch_Y.size(0)
            
            del batch_X, batch_Y, Y_hat

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples

    return avg_loss, avg_accuracy
