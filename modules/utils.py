"""
This module is designed as a repository for utility functions 
that streamline and facilitate various aspects of programming tasks. 
Currently, it includes functionalities for organizing model training 
checkpoints, with the potential for future expansion to include 
additional utilities that aid in file management, data processing, 
and other programming conveniences.

The aim is to provide a centralized location for reusable code snippets 
that improve workflow efficiency and project organization.
"""
import os
import time


def create_checkpoints_subdir(checkpoints_root:str,
                              model_type: str):
    """
    Creates a subdirectory within a specified root directory for storing training checkpoints. The subdirectory name
    includes the current date and time, and the model type, to ensure uniqueness and organization.

    Params:
    - checkpoints_root (str): The root directory where the checkpoint subdirectories will be created.
    - model_type (str): The type of model for which checkpoints are being saved, used in naming the subdirectory.

    The function attempts to create the new directory and handles cases where the directory already exists or other
    unexpected errors occur.

    Returns:
    - str: The path of the newly created checkpoints directory. This includes the root directory, current date and time,
           and the model type, ensuring that the path is unique and organized. This path can be used by the calling
           function to save checkpoints directly into the created subdirectory.

    Raises:
    - FileExistsError: If the directory already exists, indicating a potential risk of overwriting existing data.
    - Exception: For any other unexpected errors that may occur during the creation of the directory.
    """
    # Construct the directory name using the current date and time and the model type.
    checkpoints_dir = os.path.join(checkpoints_root, f"{time.strftime('%Y-%m-%d_%H-%M')}_{model_type}")

    try:
        # Attempt to create the directory.
        os.makedirs(checkpoints_dir)
        print(f"The directory {checkpoints_dir} was created successfuly.")
    except FileExistsError:
        # Handle the case where the directory already exists, warning the user to avoid overwriting the existing data.
        print("Cannot initiate training in an existing directory. "
              "This directory may contain files from previous training sessions, risking data structure corruption. "
              "Please verify the environmental variables and all specified paths before attempting another run.")
        raise
    except Exception as e:
        # Handle any other unexpected errors that may occur during directory creation.
        print(f"An unexpected error occurred while creating the directory {checkpoints_dir}: {str(e)}")
        raise

    return checkpoints_dir