"""
The `optimizer_factory` module is integral to the model training process, providing a flexible and dynamic way to instantiate optimizer instances 
based on the provided model parameters and configuration options. 
It supports the initialization of various optimizer types through a unified interface, 
enabling easy integration with different training pipelines and models. 
This module supports initializing optimizers directly from parameters or loading state from a checkpoint, 
thereby accommodating both fresh training sessions and the continuation of paused training processes.

The main entry point of this module is the `optimizer_factory` function, which dynamically selects and initializes the appropriate optimizer class based on the input parameters. 
This approach simplifies the optimizer selection process, enhancing the modularity and flexibility of the training setup.
"""

from typing import Optional, Iterable
import importlib
import torch

def optimizer_factory(model_parameters: Iterable[torch.nn.parameter.Parameter],
                      optimizer_init_params: Optional[dict] = None,
                      checkpoint: Optional[dict] = None):
        """
        Dynamically creates and returns an optimizer instance based on the provided model parameters and initialization parameters.

        This function supports initializing a new optimizer instance or loading an optimizer's state from a checkpoint. It leverages
        dynamic module import and class instantiation to create the optimizer, thus providing flexibility in the choice of optimizer
        without hardcoding specific optimizer classes.

        Params:
            model_parameters (Iterable[torch.nn.parameter.Parameter]): The parameters of the model to optimize.
            optimizer_init_params (Optional[dict]): A dictionary containing initialization parameters for the optimizer,
                including the module and class name of the optimizer to instantiate. If None and no checkpoint is provided,
                a ValueError is raised.
            checkpoint (Optional[dict]): A dictionary containing checkpoint data to derive the optimizer_state_dict from. If provided,
                the optimizer's state is loaded from this checkpoint.

        Returns:
            An instance of the optimizer loaded with the specified model parameters and optionally with state from a checkpoint.

        Raises:
            ValueError: If `optimizer_init_params` is None and no checkpoint is provided, indicating that there are not enough
                details to initialize the optimizer.

        The `optimizer_init_params` dictionary should include:
             - `optimizer_type_info`: A sub-dictionary with keys `optimizer_module_name` (e.g., "torch.optim") and `optimizer_class_name` (e.g., "SGD") 
                specifying the optimizer's module and class.

             - Additional keys corresponding to optimizer initialization arguments (e.g., `lr` for learning rate, `momentum`).

            Example structure without a checkpoint:
            ```
            optimizer_init_params = {
                'optimizer_type_info': {
                    'optimizer_module_name': "torch.optim",
                    'optimizer_class_name': "SGD"
                },
                'lr': 0.001,
                'momentum': 0.9
            }
            ```
            If a checkpoint is provided, `optimizer_init_params` is extracted from `training_init_params` within the checkpoint, 
            facilitating the continuation of training with the previously used optimizer settings.

            Example structure with a checkpoint:
            ```
            checkpoint = {
                'model_state_dict': ...,
                'optimizer_state_dict': ...,
                'epoch': ...

                'training_init_params': {
                    'training_hyperparams': ...,
                    'model_init_params': ...,

                    'optimizer_init_params': {
                        'optimizer_type_info': {
                            'optimizer_module_name': "torch.optim",
                            'optimizer_class_name': "SGD"
                        },
                        'lr': 0.001,
                        'momentum': 0.9
                    }
                },
            }
            ```

        """
        # Initialize the optimizer without a checkpoint.
        if checkpoint is None:
            if optimizer_init_params is None:
                # Ensure optimizer initialization parameters are provided if no checkpoint is given.
                raise ValueError("optimizer_init_params must be provided if checkpoint is None")
            
            # Create and return a new optimizer instance with provided parameters.
            return _get_optimizer(model_parameters=model_parameters,
                                  optimizer_init_params=optimizer_init_params)
        else:
            # Extract training_init_params dictionary from the checkpoint dict.
            training_init_params = checkpoint['training_init_params']

            # Extract optimizer_init_params dictionary from the training_init_params dict.
            optimizer_init_params = training_init_params['optimizer_init_params']
            # Create a new optimizer instance with provided parameters.
            optimizer = _get_optimizer(model_parameters=model_parameters,
                                       optimizer_init_params=optimizer_init_params)
            
            # Load the optimizer state from the checkpoint for continuing training.
            optimizer.load_state_dict(state_dict=checkpoint['optimizer_state_dict'])

            # Create and return a new optimizer instance with provided parameters.
            return optimizer


def _get_optimizer(model_parameters: Iterable[torch.nn.parameter.Parameter],
                   optimizer_init_params:dict):
    """
    Internal helper function to instantiate the optimizer based on provided parameters.

    Params:
        model_parameters (Iterable[torch.nn.parameter.Parameter]): The parameters of the model that the optimizer will update.
        optimizer_init_params (dict): A dictionary containing the module and class name of the optimizer, along with any
            additional initialization arguments.

    Returns:
        An instantiated optimizer object ready for use in training.
    """
    # Extract optimizer_type_info dictionary from the optimizer_init_params dict.
    optimizer_type_info = optimizer_init_params['optimizer_type_info']

    OptimizerClass = _get_optimizer_class(optimizer_type_info=optimizer_type_info)

    # Remove the optimizer_type_info from the optimizer_init_params dict
    # to pass the rest as **kwargs to the optimizer constructor.
    del optimizer_init_params['optimizer_type_info']

    # Instantiate the optimizer with the remaining initialization parameters.
    optimizer = OptimizerClass(params=model_parameters, **optimizer_init_params)

    return optimizer # Return the newly created optimizer instance.

def _get_optimizer_class(optimizer_type_info: dict):
    
    """
    Dynamically imports and retrieves an optimizer class using information provided in `optimizer_type_info`.

    Parameters:
        optimizer_type_info (dict): Contains 'optimizer_module_name' and 'optimizer_class_name' keys, specifying the
        module and class name of the optimizer to be imported and used.

    Returns:
        The optimizer class specified by `optimizer_type_info`.
    """
    # Extract the module and class name from the optimizer_type_info dictionary.
    optimizer_module_name = optimizer_type_info['optimizer_module_name']
    optimizer_class_name = optimizer_type_info['optimizer_class_name']

    # Dynamically import the optimizer module.
    module = importlib.import_module(optimizer_module_name)

    # Get the optimizer class from the imported module.
    OptimizerClass = getattr(module, optimizer_class_name)

    return OptimizerClass