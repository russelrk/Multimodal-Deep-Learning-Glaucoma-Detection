import os
from torchvision import transforms


class ModelParams:
    def __init__(
        self, 
        model, 
        model_name="model", 
        weights=None, 
        parent_dir="", 
        batch_size=128, 
        epochs=10, 
        learn_rate=1e-5, 
        image_size=224, 
        num_workers=16, 
        cross_val=0, 
        k_fold=10, 
        inner_k_fold=10,
        random_state=None,
        file_type="OR", 
        train=False, 
        train_data=None,
        train_transforms=None, 
        eval_transforms=None, 
    ):
        """Initialize the model parameters with default values.
        Args:
            model: The machine learning model to be used.
            model_name (str, optional): The name of the model. Defaults to "model".
            weights (optional): The weights to be used for the model initialization. Defaults to None.
            parent_dir (str, optional): The parent directory for storing output and results. Defaults to the current directory.
            batch_size (int, optional): The batch size for training the model. Defaults to 128.
            epochs (int, optional): The number of epochs for training the model. Defaults to 10.
            learn_rate (float, optional): The learning rate for the model training. Defaults to 1e-5.
            image_size (int, optional): The size of the images to be used in the model. Defaults to 224.
            num_workers (int, optional): The number of worker threads to be used in data loading. Defaults to 16.
            cross_val (int, optional): Current cross-validation fod number . Defaults to 0.
            k_fold (int, optional): The number of folds in k-fold cross-validation. Defaults to 10.
            file_type (str, optional): The file type to be used for data reading. Defaults to "OR".
            train (bool, optional): Flag to indicate if the model is in training mode. Defaults to False.
            train_transforms (optional): The transformations to be applied during training. Defaults to None.
            eval_transforms (optional): The transformations to be applied during evaluation. Defaults to None.
            data_file (optional): The file containing the data to be used. Defaults to None.
            save_fig_path (str, optional): The path where output figures during the model's execution will be stored. 
            save_results_path (str, optional): The directory to store the prediction results generated during the model's execution. 
            save_model_path (str, optional): The directory to store the trained model files. 
            checkpoint_path_loss (str, optional): The filepath to save the checkpoint file which keeps track of the model with the least loss during training.
            checkpoint_path_auc (str, optional): The filepath to save the checkpoint file which keeps track of the model with the highest AUC during validation. 
            final_weight_path (str, optional): The filepath to save the final weights of the trained model after the completion of all epochs. 
            
        """
      
        self.model = model
        self.model_name = model_name
        self.parent_dir = parent_dir
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LR = learn_rate
        self.IMG_SIZE = image_size
        self.NUM_WORKERS = num_workers
        self.cross_val = cross_val
        self.k_fold = k_fold
        self.file_type = file_type
        self.train = train
        self.train_data = train_data
        self.weights = weights
        self.random_state = random_state
        

        # Define default training and evaluation transformations
        self._default_transforms(train_transforms, eval_transforms)

        # Setup directories to store output figures, prediction results, and model checkpoints
        self._setup_directories()

    def _default_transforms(self, train_transforms, eval_transforms):
        """Define the default transformations for training and evaluation.

        Args:
            train_transforms: Transformations to be applied during training.
            eval_transforms: Transformations to be applied during evaluation.
        """
        TrainTransforms = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.RandomAutocontrast(),
            transforms.RandomAffine(degrees=(-25, 25), translate=(0.2, 0.3), shear=(5, 5)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        EvalTransforms = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_transforms = train_transforms if train_transforms else TrainTransforms
        self.eval_transforms = eval_transforms if eval_transforms else EvalTransforms

    def _setup_directories(self):
        """Setup directories to store outputs and checkpoint files."""
        self._create_directory(os.path.join(self.parent_dir, 'output_figures'))
        self._create_directory(os.path.join(self.parent_dir, 'prediction_results'))

        self.save_fig_path = self._create_directory(os.path.join(self.parent_dir, f'output_figures/{self.model_name}'))
        self.save_results_path = self._create_directory(os.path.join(self.parent_dir, f'prediction_results/{self.model_name}'))
        self.save_model_path = self._create_directory(os.path.join(self.parent_dir, 'saved_models'))
        self._create_directory(os.path.join(self.save_model_path, f"{self.model_name}"))

        self.checkpoint_path_loss = os.path.join(self.save_model_path, f"{self.model_name}/check_point_loss.pt")
        self.checkpoint_path_auc = os.path.join(self.save_model_path, f"{self.model_name}/check_point_auc.pt")
        self.final_weight_path = os.path.join(self.save_model_path, f"{self.model_name}/final_weight.pt")

    def _create_directory(self, path):
        """Create a directory if it does not exist.
        Args:
            path (str): The path of the directory to be created.
        
        Returns:
            str: The path of the directory.
        """
      
        if not os.path.exists(path):
            os.mkdir(path)
        return path
