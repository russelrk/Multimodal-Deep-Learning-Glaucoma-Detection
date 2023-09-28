import argparse
from torch.nn.modules.loss import BCEWithLogitsLoss
from ..shared.data_preprocess import read_split_data, get_data_loader
from ..shared.models import get_model
from ..shared.model_params import ModelParams
from ..shared.train_validate import train_model


def main(args):
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
        train_transforms=None,
        eval_transforms=None,
        data_file=None
    )

    df_train, df_valid, df_test = read_split_data(
        k_fold=args.k_fold,
        cross_val=args.cross_val,
        random_state=args.random_state,
        inner_k_fold=args.inner_k_fold,
        data_file=args.train_data
    )

    train_loader, trainSteps, class_weights = get_data_loader(
        df_train,
        split_type = 'train',
        model_params = modelParams,
        data_type='cfp', 
        fid = False
    )
    
    valid_loader, valSteps = get_data_loader(
        df_valid,
        split_type = 'valid',
        model_params = modelParams,
        data_type='cfp', 
        fid = False
    )
    
    test_loader, testSteps = get_data_loader(
        df_test,
        split_type = 'test',
        model_params = modelParams,
        data_type='cfp', 
        fid = False
    )
    
    model = get_model(
        model_name = modelParams.model,
        get_default_weights = modelParams.weights,
        freeze = False,
        num_classes = 1,
    )

    model.to(args.device)

    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=modelParams.LR)

    model = train_model(
        model,     
        valid_loader, 
        test_loader, 
        test_loader, 
        device=args.device, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        model_params=modelParams
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training a classification model.")
    parser.add_argument("--model", default="resnet18", help="Name of the model.")
    parser.add_argument("--model_name", default="model18", help="Name for the model.")
    parser.add_argument("--weights", action='store_true', help="Use pretrained weights.")
    parser.add_argument("--parent_dir", dfault='', help="parent directory of the file.")
    parser.add_argument("--batch_size", default=64, type=int, help="defaults batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="nuber of epochs to run.")
    parser.add_argument("--learn_rate", default=0.001, type=float, help="learning rate for training the model")
    parser.add_argument("--image_size", default=224, type=int, help="size of the image for training")
    parser.add_argument("--num_workers", default=4, type=int, help="number of worker")
    parser.add_argument("--cross_val", default=0, type=int, help="cross_val to run")
    parser.add_argument("--k_fold", default=10, type=int, help="cross validatio fold number")
    parser.add_argument("--file_type", default='OR', help="file_type optional")
    parser.add_argument("--train", action='store_true', help="to train or not")
    parser.add_argument("--train_data", default="image_path_train.tsv", help="Path to train data.")
    parser.add_argument("--device", default="cuda", help="Device to use for training, e.g., cuda or cpu.")

    args = parser.parse_args()
    main(args)

# python your_script_name.py --model resnet18 --weights --train_data /path/to/train_data.tsv

