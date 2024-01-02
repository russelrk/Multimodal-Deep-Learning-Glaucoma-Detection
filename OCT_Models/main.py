import argparse
import torch
from torch.nn.modules.loss import BCEWithLogitsLoss

import sys
sys.path.append('..')  # Add the parent directory to the sys.path

from shared.data_preprocess import read_split_data, get_data_loader
from shared.models import get_model
from shared.model_params import ModelParams
from shared.train_and_validate import train_model

def main(modelParams):

    # Read and split the data for training and validation
    df_train, df_valid, df_test = read_split_data(
        k_fold=modelParams.k_fold,
        cross_val=modelParams.cross_val,
        random_state=modelParams.random_state,
        inner_k_fold=modelParams.k_fold,
        data_file=modelParams.train_data
    )
    
    # Get data loaders for training, validation, and test datasets
    train_loader, trainSteps, class_weights = get_data_loader(
        df_train,
        split_type='train',
        model_params=modelParams,
        data_type='oct',
        fid=False
    )
    
    valid_loader, valSteps = get_data_loader(
        df_valid,
        split_type='valid',
        model_params=modelParams,
        data_type='oct',
        fid=False
    )
    
    test_loader, testSteps = get_data_loader(
        df_test,
        split_type='test',
        model_params=modelParams,
        data_type='oct',
        fid=False
    )
    
    # Get the model architecture
    model = get_model(
        model_name=modelParams.model,
        get_default_weights=modelParams.weights,
        freeze=False,
        num_classes=1,
    )

    # Move the model to the specified device (e.g., GPU)
    model.to(args.device)

    # Define loss function and optimizer
    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=modelParams.LR)

    print('Train Model', flush = True)
    # Train the model
    model = train_model(
        model,
        valid_loader,
        test_loader,
        test_loader,
        device=args.device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        model_params=modelParams, 
    )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training a classification model.")
    parser.add_argument("--model", default="resnet18", help="Name of the model.")
    parser.add_argument("--model_name", default="model18", help="Name for the model.")
    parser.add_argument("--weights", action='store_true', default=False, help="Use pretrained weights.")
    parser.add_argument("--parent_dir", default='', help="Parent directory of the file.")
    parser.add_argument("--batch_size", default=64, type=int, help="Defaults batch size.")
    parser.add_argument("--epochs", default=2, type=int, help="Number of epochs to run.")
    parser.add_argument("--learn_rate", default=0.001, type=float, help="Learning rate for training the model.")
    parser.add_argument("--image_size", default=224, type=int, help="Size of the image for training.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of worker.")
    parser.add_argument("--cross_val", default=0, type=int, help="Cross_val to run.")
    parser.add_argument("--k_fold", default=10, type=int, help="Cross-validation fold number.")
    parser.add_argument("--random_state", default=0, type=int, help="Random state.")
    parser.add_argument("--file_type", default='OR', help="File_type optional.")
    parser.add_argument("--train", action='store_true', default=False, help="To train or not.")
    parser.add_argument("--train_data", default="image_path_train.tsv", help="Path to train data.")
    parser.add_argument("--device", default="cuda", help="Device to use for training, e.g., cuda or cpu.")


    args = parser.parse_args()
    
    # Initialize model parameters using the provided arguments
    modelParams = ModelParams(
        args.model,
        model_name=args.model_name,
        weights=args.weights,
        parent_dir=args.parent_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learn_rate=args.learn_rate,
        image_size=args.image_size,
        num_workers=args.num_workers,
        cross_val=args.cross_val,
        k_fold=args.k_fold,
        file_type=args.file_type,
        train=args.train,
        train_data=args.train_data,
        random_state=args.random_state,
        train_transforms=None,
        eval_transforms=None,
    )
    
    main(modelParams)

# Example usage:
# python your_script_name.py --model resnet18 --train_data /path/to/train_data.tsv
