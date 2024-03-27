"""
This module provides a factory function for dynamically creating optimizer instances. 
It is designed to be agnostic of the optimization library, supporting a flexible integration 
with various frameworks by specifying optimizer types and initialization parameters. 
This design facilitates easy switching between different optimizers and resuming training 
from checkpoints, enhancing usability in diverse machine learning tasks.
"""
from typing import Optional, Iterable
import importlib
import torch

def optimizer_factory(model_parameters: Iterable[torch.nn.parameter.Parameter],
                      optimizer_type_info: Optional[dict] = None,
                      optimizer_init_params: Optional[dict] = None,
                      checkpoint: Optional[dict] = None):
        """
        Creates and returns an optimizer instance based on provided parameters or a checkpoint.
        This function is framework-agnostic and can be used with various machine learning libraries.

        Parameters:
        - model_parameters (Iterable): Parameters of the model to optimize.
        - optimizer_type_info (Optional[dict]): Information about the optimizer type, including module and class name.
        - optimizer_init_params (Optional[dict]): Initialization parameters for the optimizer.
        - checkpoint (Optional[dict]): A checkpoint containing optimizer state and initialization parameters.

        Returns:
        - An instance of the requested optimizer.

        Raises:
        - ValueError: If necessary initialization parameters are missing when not loading from a checkpoint.
        """
        # Initialize the optimizer without a checkpoint.
        if checkpoint is None:
            if None in (optimizer_type_info, optimizer_init_params):
                # Ensure optimizer initialization parameters are provided if no checkpoint is given.
                raise ValueError("Both optimizer_type_info and optimizer_init_params must be provided if checkpoint is None")
            
            # Create and return a new optimizer instance with provided parameters.
            return _get_optimizer(model_parameters=model_parameters,
                                  optimizer_type_info=optimizer_type_info,
                                  optimizer_init_params=optimizer_init_params)
        else:
            # Extract training_init_params dictionary from the checkpoint dict.
            training_init_params = checkpoint['training_init_params']
            # Extract optimizer_type_info dictionary from the training_init_params dict.
            optimizer_type_info = training_init_params['optimizer_type_info']
            # Extract optimizer_init_params dictionary from the training_init_params dict.
            optimizer_init_params = training_init_params['optimizer_init_params']
            # Create a new optimizer instance with provided parameters.
            optimizer = _get_optimizer(model_parameters=model_parameters,
                                       optimizer_type_info=optimizer_type_info,
                                       optimizer_init_params=optimizer_init_params)
            
            # Load the optimizer state from the checkpoint for continuing training.
            optimizer.load_state_dict(state_dict=checkpoint['optimizer_state_dict'])

            # Create and return a new optimizer instance with provided parameters.
            return optimizer


def _get_optimizer(model_parameters: Iterable[torch.nn.parameter.Parameter],
                   optimizer_type_info: dict,
                   optimizer_init_params:dict):
    """
    Internal helper function to instantiate an optimizer.
    It dynamically imports and creates an optimizer instance based on the provided type and parameters.

    Parameters:
    - model_parameters (Iterable): Parameters of the model to optimize.
    - optimizer_type_info (dict): Information about the optimizer type, including module and class name.
    - optimizer_init_params (dict): Initialization parameters for the optimizer.

    Returns:
    - An instantiated optimizer.
    """
    # Extract the module and class name from the optimizer_type_info dictionary.
    optimizer_module_name = optimizer_type_info['optimizer_module_name']
    optimizer_class_name = optimizer_type_info['optimizer_class_name']

    # Dynamically import the optimizer module.
    module = importlib.import_module(optimizer_module_name)

    # Get the optimizer class from the imported module.
    OptimizerClass = getattr(module, optimizer_class_name)

    # Instantiate the optimizer with the remaining initialization parameters.
    optimizer = OptimizerClass(params=model_parameters, **optimizer_init_params)

    return optimizer # Return the newly created optimizer instance.