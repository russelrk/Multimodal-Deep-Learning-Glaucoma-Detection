import logging
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Any, Dict
from torch.nn import Module
from torch.utils.data import DataLoader
from .model_params import ModelParams

# Configure logging
logging.basicConfig(level=logging.INFO)


def evaluate(model: Module, data_loader: DataLoader, split: str, device: str, model_params: ModelParams) -> None:
    """Evaluate model on a data loader and save results to a file.

    Args:
        model (Module): The PyTorch model to evaluate.
        data_loader (DataLoader): The DataLoader containing the evaluation dataset.
        split (str): The split name to use in logging and output file names.
        device (str): The device (CPU or GPU) to use for evaluation.
        model_params (dict): Dictionary containing all necessary parameters for model evaluation.

    Returns:
        None
    """
    with torch.no_grad():
        model.eval()

        preds, labels, fids = [], [], []
        for x, y, fid in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).to(torch.float32)
            y = y.unsqueeze(1).to(torch.float32)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
            fids.extend(fid)

        preds = [i[0] for i in preds]
        labels = [i[0] for i in labels]

        logging.info(f"AUC score obtained from {split} dataset: {roc_auc_score(labels, preds)}")

        df = pd.DataFrame({'FID': fids, 'labels': labels, 'preds': preds})
        df.to_csv(os.path.join(model_params['save_results_path'], f'pred_results_{split}.tsv'), sep='\t', index=False)
