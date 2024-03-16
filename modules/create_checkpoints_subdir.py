"""
This module provides functionality for creating a directory for model checkpoints based on the current date and time,
and the type of model being trained. It is designed to ensure that each training session's checkpoints are stored
in a unique directory to prevent data from being overwritten or mixed with data from other training sessions.
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

    No return value.
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
    except Exception as e:
        # Handle any other unexpected errors that may occur during directory creation.
        print(f"An unexpected error occurred while creating the directory {checkpoints_dir}: {str(e)}")