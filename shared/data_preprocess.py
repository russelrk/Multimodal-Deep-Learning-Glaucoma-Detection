from typing import List, Callable, Union, Optional, Tuple
import pandas as pd
import numpy as np
import logging

def read_split_data(
    k_fold: int = 5, 
    cross_val: int = 0, 
    random_state: int = 10, 
    inner_k_fold: int = 10, 
    cross_val_inner: Optional[int] = None, 
    slice_no: Optional[int] = None, 
    data_file: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads and splits the dataset into label stratafied training, validation, and test sets based on k-fold cross-validation. 
    The code also ensures a dataset with same unique identifier belongs to the same data-split. 
    This is function is particularly helpful for unbalanced dataset where tha same object has multiple different input features. 

    

    Parameters:
    - k_fold (int): Number of folds for outer cross-validation. Defaults to 5.
    - cross_val (int): Current fold number for outer cross-validation. Defaults to 0.
    - random_state (int): Random seed for shuffling the dataset. Defaults to 10.
    - inner_k_fold (int): Number of folds for inner cross-validation. Defaults to 10.
    - cross_val_inner (int, optional): Current fold number for inner cross-validation. Defaults to None.
    - slice_no (int, optional): Slice number if applicable. Defaults to None.
    - data_file (str, optional): Path to the data file. Defaults to the hardcoded path.
    
    Returns:
    - tuple: Training, validation, and test DataFrames.
    """
    
    if data_file is None:
        """
        data file will be in the tsv file format, containing at least 3 columnss. 
        columns: 
            ID: unique identifier
            label: data label 
            img_path: path of each image dataset
        """
      
        data_file = 'file_path_labels.tsv'
    
    logging.info(f"[INFO]: File Loaded: {data_file}")

    try:
        df = pd.read_csv(data_file, sep='\t', low_memory=False)
        for i in ["FID", "IID"]:
            if i in df.columns:
                df = df.rename(columns={i: "ID"})
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        raise

    if cross_val_inner is None:
        cross_val_inner = 0 if cross_val == k_fold - 1 else cross_val

    if k_fold is not None:
        df_pos = df.loc[df['label'] == 1]
        df_neg = df.loc[df['label'] == 0]

        pos_ids = list(df_pos['ID'].unique())
        neg_ids = list(df_neg['ID'].unique())

        pos_splits = np.array_split(pos_ids, k_fold)
        neg_splits = np.array_split(neg_ids, k_fold)

        test_ids = list(pos_splits.pop(cross_val)) + list(neg_splits.pop(cross_val))
        val_ids = list(pos_splits.pop(cross_val_inner)) + list(neg_splits.pop(cross_val_inner))
        
        train_ids = [j for sub in pos_splits for j in sub] + [j for sub in neg_splits for j in sub]

        df_train = df.loc[df['ID'].isin(train_ids)].copy().reset_index(drop=True)
        df_val = df.loc[df['ID'].isin(val_ids)].copy().reset_index(drop=True)
        df_test = df.loc[df['ID'].isin(test_ids)].copy().reset_index(drop=True)
        
        return df_train, df_val, df_test

    else:
        return df



from torch.utils.data import Dataset
from PIL import Image
import random

class Fundus_Dataset(Dataset):
    def __init__(
        self, 
        img_path: List[str], 
        labels: List[int], 
        fids: List[str], 
        img_size: int = 224, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None, 
        fid: bool = False
    ) -> None:
        self.fid = fid
        self.img_labels = labels
        self.img_dir = img_path
        self.img_fid = fids
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        image = Image.open(self.img_dir[idx]) #read_image(self.img_dir[idx]).to(torch.float32)
        label = self.img_labels[idx]
        fids = self.img_fid[idx]
            
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)

        if self.fid:
            return image, label, fids
        else:
            return image, label

class OCT_Dataset(Dataset):
    def __init__(
        self, 
        img_path: List[str], 
        labels: List[int], 
        fids: List[str], 
        img_size: int = 224, 
        volume: bool = False,
        volume_step: int = 1,
        transform: Optional[Callable] = None, 
        augment: Optional[bool] = None,
        target_transform: Optional[Callable] = None, 
        fid: bool = False
    ) -> None:
        
        """
        Initialize the OCT_Dataset class.

        Parameters:
        - img_path (list of str): Paths to the images.
        - labels (list): List of labels corresponding to the images.
        - fids (list): List of file IDs corresponding to the images.
        - img_size (int, optional): The size to which images should be resized. Defaults to 224.
        - volume (bool, optional): 
        - transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        - target_transform (callable, optional): A function/transform to apply to the labels. Defaults to None.
        - fid (bool, optional): Whether to return the file ID along with the image and label. Defaults to False. Needed during model evaluation to keep tract of each results tied to each FID. 
        """
        self.img_fids = fids
        self.img_labels = labels
        self.img_dir = img_path
        self.volume = volume
        self.volume_step = volume_step
        self.transform = transform
        self.augment = augment
        self.target_transform = target_transform
        self.fid = fid
        self.img_size = img_size

    def __len__(self):
        """
        Returns:
        int: The number of samples in the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Fetch an item from the dataset.

        Parameters:
        - idx (int): The index of the item to fetch.

        Returns:
        - image: The image corresponding to the index.
        - label: The label corresponding to the image.
        - fid (optional): The file ID corresponding to the image.
        """
        if self.volume:
            """
            load volume image 
            """
            image = []
            p1 = 1 if random.random() >= 0.5 else 0
            p2 = 1 if random.random() >= 0.5 else 0
            rot = random.randint(-25,25)
            for i in range(1, 129, self.volume_step):
                temp = Image.open(self.img_dir[idx]+f"_{i}.png")
                if self.augment:
                    temp = transforms.functional.hflip(temp) if p1 else temp
                    temp = transforms.functional.vflip(temp) if p2 else temp
                    temp = transforms.functional.rotate(temp, rot)
                        
                if self.transform:
                    temp = self.transform(temp)

                if temp:
                    image.append(temp)
    
            image = (torch.stack(image, dim = 1))

        else:
            # load each B-Scan
            image = Image.open(self.img_dir[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            

        label = self.img_labels[idx]

            
        if self.target_transform:
            label = self.target_transform(label)
          
        if self.fid:
            fid = self.img_fids[idx]
            return image, label, fid
        else:
            return image, label



from torch.utils.data import DataLoader
# from .model_params import ModelParams


def get_data_loader(
    df: pd.DataFrame,
    split_type: str,
    model_params: ModelParams,
    data_type: str, 
    fid: bool = False
) -> Union[Tuple[DataLoader, int, float], Tuple[DataLoader, int]]:
    """
    Creates a data loader from the provided DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the dataset information.
    - split_type (str): The type of data split ("train", "validation", "test").
    - model_params (ModelParams): A class to initilize all necessary constant to run and train the DL model
    - transforms (callable, optional): The transformations to be applied to the dataset.
    - fid (bool, optional): Whether to include file ID in the dataset. Defaults to False.
    
    Returns:
    - DataLoader: The DataLoader object.
    - int: The number of steps required to go through the dataset.
    - float (only for "train" split_type): The class weight for the positive class.
    """
    
    img_paths = df['img_path'].tolist()
    labels = df['label'].tolist()
    fids = df['ID'].tolist()
    
    if data_type == 'oct':
        dataset = OCT_Dataset(
            img_paths, 
            labels, 
            fids, 
            img_size=model_params.IMG_SIZE, 
            transform=model_params.train_transforms, 
            fid=fid
        )
    else:
        dataset = Fundus_Dataset(
            img_paths, 
            labels, 
            fids, 
            img_size=model_params.IMG_SIZE, 
            transform=model_params.train_transforms, 
            fid=fid
        )  

    data_loader = DataLoader(
        dataset, 
        shuffle=True, 
        batch_size=model_params.BATCH_SIZE, 
        num_workers=model_params.NUM_WORKERS, 
        pin_memory=True
    )

    steps = len(data_loader.dataset) // model_params.BATCH_SIZE
    
    print(f"[INFO]: Total number of {split_type} dataset: ", len(data_loader.dataset), flush=True)
    
    if split_type == "train":
        class_counts = df['label'].value_counts()
        class_weight = class_counts[0] / class_counts[1]
        return data_loader, steps, class_weight
    else:
        return data_loader, steps

