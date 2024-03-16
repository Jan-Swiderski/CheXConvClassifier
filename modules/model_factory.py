"""
This module serves as a central hub for dynamically creating and configuring machine learning models based on the given model type. 
It utilizes a mapping between model type identifiers and module names that contain the actual model initialization logic. 
Through this mapping, the `model_factory` function dynamically imports the necessary module and invokes a standard `get_model` function 
defined within, passing any initialization parameters or checkpoints if provided. This approach facilitates flexibility and ease of 
expansion for the project by abstracting model instantiation into a unified interface, which can be extended simply by adding new model 
modules and updating the `models_map`.
"""
from typing import Optional
import torch
from .get_classifier import get_model as get_classifier
from .get_mobilenet_v3 import get_model as get_mobilenet_v3

models_map = {'classifier': get_classifier,
              'mobilenet_v3_small': get_mobilenet_v3,
              'mobilenet_v3_large': get_mobilenet_v3}


def model_factory(model_init_params: Optional[dict] = None,
                  checkpoint: Optional[dict] = None):
    """
    Dynamically creates and returns a model instance based on the specified type, initialization parameters, and optional checkpoint.
    
    This function first validates the model type against a predefined mapping, then dynamically imports the appropriate module
    containing a `get_model` function. It initializes the model either from scratch using `model_init_params` or from a given
    checkpoint, and finally moves the model to the available device (CPU or CUDA).

    Params:
        model_init_params (Optional[dict]): A dictionary of parameters required for model initialization, defaults to None.
        checkpoint (Optional[dict]): A dictionary containing a model checkpoint for initializing the model, defaults to None.
        
    Returns:
        A PyTorch model instance configured and possibly loaded with checkpoint data.

    Raises:
        ValueError: If the model type is not supported or necessary initialization parameters are missing.
    """
    
    # Determine the computing device (CUDA if available, else CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model without a checkpoint.
    if checkpoint is None:

        if model_init_params is None:
            # Ensure model initialization parameters are provided if no checkpoint is given.
            raise ValueError("model_init_params must be provided if checkpoint is None")
        
        model_type = model_init_params['model_type']

        # Validate if the requested model type is supported.
        _validate_model_type(model_type)
        
        # Instantiate the model using provided initialization parameters.
        model = models_map[model_type](model_init_params)
        model.to(device) # Move the model to the appropriate device.

        return model
        
    else:
        # Initialize model with a checkpoint.

        # Extracting training initialization parameters from the checkpoint.
        trainig_init_params = checkpoint['training_init_params']

        # Extracting model initialization parameters from the training initialization parameters stored in the checkpoint.
        model_init_params = trainig_init_params['model_init_params']

        model_type = model_init_params['model_type']
        
        # Validate if the requested model type is supported.
        _validate_model_type(model_type)

        # Instantiate the model using initialization parameters provided in the checkpoint.
        # model = module.get_model(model_init_params)
        model = models_map[model_type](model_init_params)

        # Load model state from the checkpoint.
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        
        model.to(device)  # Move the model to the appropriate device.

        return model
    

def _validate_model_type(model_type):
    """
    Internal helper function to validate if proper model_type was given
    """
    # Validate if the requested model type is supported.
    if model_type not in models_map:
        raise ValueError(f"Model type of {model_type} is not supported")

