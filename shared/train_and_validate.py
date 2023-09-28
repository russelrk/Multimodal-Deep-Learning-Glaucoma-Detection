from typing import Tuple, List, Callable, Any
import time
import torch
from sklearn.metrics import roc_auc_score
from .model_params import ModelParams

def make_train_step(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   loss_fn: Callable) -> Callable:
    """
    Creates a training step function.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to train
    - optimizer (torch.optim.Optimizer): The optimizer to use during training
    - loss_fn (Callable): The loss function

    Returns:
    - train_step (Callable): A function that takes a batch of data and labels, and performs a single step of training
    """
    def train_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        model.train()  # Set model to training mode
        yhat = model(x)  # Make prediction
        loss = loss_fn(yhat, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset gradients
        return loss
    return train_step


def evaluate_model(data_loader: torch.utils.data.DataLoader, 
                   model: torch.nn.Module, 
                   device: torch.device, 
                   loss_fn: Callable) -> Tuple[float, List[float], List[float]]:
    """
    Evaluates the model on a given dataset.

    Parameters:
    - data_loader (torch.utils.data.DataLoader): The data loader
    - model (torch.nn.Module): The PyTorch model to evaluate
    - device (torch.device): The device (CPU or GPU) where to perform computations
    - loss_fn (Callable): The loss function

    Returns:
    - cum_loss (float): The cumulative loss
    - preds (List[float]): The list of predictions
    - labels (List[float]): The list of ground truth labels
    """
    cum_loss = 0
    preds = []
    labels = []

    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.unsqueeze(1).float().to(device)
        yhat = model(x_batch)
        
        preds.extend(yhat.cpu().numpy())
        labels.extend(y_batch.cpu().numpy())
        
        val_loss = loss_fn(yhat, y_batch)
        cum_loss += val_loss.item() / len(data_loader)
    return cum_loss, preds, labels




def train_model(model: torch.nn.Module, 
                train_data_loader: torch.utils.data.DataLoader, 
                val_data_loader: torch.utils.data.DataLoader, 
                test_data_loader: torch.utils.data.DataLoader, 
                device: torch.device, 
                loss_fn: Callable, 
                optimizer: torch.optim.Optimizer, 
                model_params: Any) -> None:

    train_step = make_train_step(model, optimizer, loss_fn)
    
    losses = []
    val_losses = []
    
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_test_auc = []

    n_epochs = model_params.EPOCHS
    early_stopping_tolerance = 3
    early_stopping_threshold = 0.03
    early_stopping_counter = 0

    for epoch in range(n_epochs):
        time_st = time.time()
        epoch_loss = 0.0

        model.train()
        for i, (x_batch, y_batch) in enumerate(train_data_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.unsqueeze(1).float().to(device)
            
            loss = train_step(x_batch, y_batch)
            epoch_loss += loss.item() / len(train_data_loader)
            losses.append(loss.item())

        epoch_train_losses.append(epoch_loss)
        time_en = (time.time() - time_st) / 60.0

        print(f'\nEpoch : {epoch + 1}, train loss : {epoch_loss:.4f}, time/Epoch: {time_en:.2f}', flush=True)

        with torch.no_grad():
            model.eval()
            cum_loss, preds, labels = evaluate_model(val_data_loader, model, device, loss_fn)
            
            val_auc = roc_auc_score(np.concatenate(labels), np.concatenate(preds))
            epoch_test_losses.append(cum_loss)
            epoch_test_auc.append(val_auc)
            
            print(f'Epoch : {epoch + 1}, val loss : {cum_loss:.4f}, AUC {val_auc:.4f}', flush=True)

        best_loss = min(epoch_test_losses)
        
        if cum_loss <= best_loss:
            print("     [INFO]: Saving the best loss model ...", flush=True)
            torch.save(model.state_dict(), model_params.checkpoint_path_loss)
        
        if val_auc >= max(epoch_test_auc):
            print("     [INFO]: Saving the best auc model ...", flush=True)
            torch.save(model.state_dict(), model_params.checkpoint_path_auc)
        
        if cum_loss > best_loss:
            early_stopping_counter += 1
    
        if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
            print("\nTerminating: early stopping", flush=True)
            break 

        with torch.no_grad():
            model.eval()
            cum_loss, preds, labels = evaluate_model(test_data_loader, model, device, loss_fn)
            
            test_auc = roc_auc_score(np.concatenate(labels), np.concatenate(preds))
            
            print(f'Epoch : {epoch + 1}, test loss : {cum_loss:.4f}, AUC {test_auc:.4f}', flush=True)  

    print("[INFO]: Finished Training ...", flush=True)
    print("[INFO]: Saving the final model ...", flush=True)
    torch.save(model.state_dict(), model_params.final_weight_path)

 


