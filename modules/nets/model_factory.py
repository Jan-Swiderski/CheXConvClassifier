"""
The `model_factory` module acts as a flexible, centralized hub for creating and configuring different machine learning models within the project.
It abstracts model initialization into a single interface, enabling easy model switching and configuration.

Through a `models_map` dictionary, this module maps model type identifiers to their respective module paths, facilitating dynamic model instantiation. 
It supports the dynamic import and invocation of a standardized `get_model` function across various model modules,
allowing for initialization with predefined parameters and optional checkpoint loading.

The `model_factory` function is the core of this module, capable of handling model creation based on provided `model_type` 
and optional initialization parameters or checkpoints. 
Each model comes with a sensible set of default parameters, ensuring that specifying only the `model_type` is sufficient 
for straightforward model initialization. This feature, along with the standardized function naming (`get_model`), 
enhances the system's modularity and scalability, making it straightforward to extend with new model types.

Main Features:
- Simplifies model selection and instantiation across diverse architectures.
- Supports default and custom initialization parameters, plus checkpoint loading for model flexibility.
- Modular design facilitates easy addition and integration of new model types.
"""
import importlib
from typing import Optional
import torch

models_map = {'chex_conv_classifier': "modules.nets.chex_conv_classifier",
              'mobilenet_v3_small': "modules.nets.get_mobilenet_v3",
              'mobilenet_v3_large': "modules.nets.get_mobilenet_v3"}


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

        module = importlib.import_module(models_map[model_type])
        model = module.get_model(**model_init_params)
        model.to(device) # Move the model to the appropriate device.
        return model     
    else:
        # Initialize model with a checkpoint.

        # Extracting training initialization parameters from the checkpoint.
        training_init_params = checkpoint['training_init_params']

        # Extracting model initialization parameters from the training initialization parameters stored in the checkpoint.
        model_init_params = training_init_params['model_init_params']

        model_type = model_init_params['model_type']
        
        # Validate if the requested model type is supported.
        _validate_model_type(model_type)

        module = importlib.import_module(models_map[model_type])

        # Instantiate the model using initialization parameters provided in the checkpoint.
        model = module.get_model(**model_init_params)
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
