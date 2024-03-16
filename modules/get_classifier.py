"""
This module is designed for use with the model_factory module, providing a uniform interface for model initialization. 
It includes a single function, `get_model`, which follows a project-wide convention for initializing models. This approach 
allows for easy integration with the model_factory module, where the differentiation between models to be initialized 
is managed through the naming of modules containing the `get_model` functions rather than the function names themselves. 

The `get_model` function in this module is specifically for initializing a custom Classifier model based on a set of 
configuration parameters. These parameters include details for convolutional layers (kernel size, stride, and output channels) 
and the fully connected layer (output features), essential for tailoring the model to specific datasets or tasks.

By maintaining a consistent function name across different model definition modules, the model_factory can dynamically 
select and initialize models based on module names, simplifying the model selection process and enhancing the modularity of the project.
"""
from .classifier import Classifier

# Defining keys required in the initialization dictionary for model customization.
init_required_keys = ['l1_kernel_size',
                'l1_stride',
                'l1_out_chann',
                'l2_kernel_size',
                'l2_stride',
                'l2_out_chann',
                'l3_kernel_size',
                'l3_stride',
                'l3_out_chann',
                'fc_out_features',
                'im_size']

def get_model(model_init_params: dict):
    """
    Initializes and returns a Classifier model based on the provided initialization parameters.
    
    This function checks if all required configuration parameters are included in the `model_init_params` dictionary.
    If any required keys are missing, it raises a ValueError. Otherwise, it initializes a Classifier model with the
    provided parameters, which include the configuration for convolutional layers and a fully connected layer, and
    returns the initialized model.
    
    Params:
    - model_init_params (dict): A dictionary containing model initialization parameters. The expected keys and their
      meanings are as follows:
      
    Returns:
    - Classifier: An instance of the Classifier model initialized with the given parameters.
    
    Raises:
    - ValueError: If any of the required keys are missing in `model_init_params`.
    """
    # Check for missing required keys in the model initialization parameters.
    missing_keys = [key for key in init_required_keys if key not in model_init_params]
    if missing_keys:
        # If there are missing keys, raise a ValueError with a detailed message
        raise ValueError(f"Missing required key(s) in model_init_params: {','.join(missing_keys)}")
    
    # Initialize the Classifier model with the given parameters
    model = Classifier(**model_init_params)

    # Return the initialized model
    return model
