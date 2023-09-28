from torch import nn
from torchvision import models
from typing import Union, Optional, Callable

def get_model(
    model_name: str = 'resnet18',
    get_default_weights: Optional[bool] = False,
    freeze: Optional[bool] = False,
    num_classes: Optional[bool] = 1,
) -> nn.Module:
    """
    Retrieves a PyTorch model with the specified characteristics. you can modify this code to further customize imported models to suit your needs. 

    Args:
    model_name (str): The name of the model to retrieve.
    get_default_weights (Callable): The pre-trained weights to load.
    freeze (bool): Whether to freeze the weights of the model layers.
    num_classes (bool): number of classes in the classification problem default to 1

    Returns:
    nn.Module: The requested model.
    """
    
    def freeze_model(model: nn.Module) -> None:
        """Freeze all parameters of the model."""
        for params in model.parameters():
            params.requires_grad_ = False
    
    print(f"[INFO]: Load weights: {get_default_weights}", flush=True)

    models_dict = {
        'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
        'densenet': ['densenet201'],
        'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b4', 'efficientnet_b7'],
        'convnext': ['convnext_tiny'],
        'swin' : ['swin_t']
    }
    
    default_weights = {
            'resnet': 'IMAGENET1K_V1',
            'densenet': 'IMAGENET1K_V1',
            'efficientnet': 'IMAGENET1K_V1',
            'convnext': 'IMAGENET1K_V1',
            'swin' : 'IMAGENET1K_V1',
    }

    model_family = next((k for k, v in models_dict.items() if model_name in v), None)
    
    if model_family is None:
        raise ValueError(f"Invalid model name: {model_name}")

    if get_default_weights:
        weights = default_weights[model_family]
    else:
        weights = None

    try:
        model = getattr(models, model_name)(weights=weights)
    except AttributeError:
        raise ValueError(f"Invalid model name: {model_name}")
      
    if model_name in models_dict['resnet']:
        nr_filters = model.fc.in_features
        model.fc = nn.Linear(nr_filters, num_classes)
        
    elif model_name in models_dict['densenet']:
        nr_filters = model.classifier.in_features
        model.classifier = nn.Linear(nr_filters, num_classes)
        
    elif model_name in models_dict['efficientnet']:
        nr_filters = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(nr_filters, num_classes)
        
    elif model_name in models_dict['convnext']:
        nr_filters = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(nr_filters, num_classes)
      
    elif model_name in models_dict['swin']:
        nr_filters = model.head.in_features
        model.head = nn.Linear(nr_filters, num_classes)

    if freeze:
        freeze_model(model)
    
    return model

