"""
Provides functionality to process and predict classes of images from a directory using a trained model.
It supports image preprocessing, prediction, and optional sorting of images into class-based directories.

Usage Convention:
-----------------
This module is integrated with `main.py` to allow command-line-based operation. It is used to predict and optionally sort images by passing the model checkpoint path, image directory path, and a sorting flag.

For example:
python main.py predict /path/to/images /path/to/model_checkpoint --sort_files

This approach ensures modular use of prediction logic separate from the command-line interface, promoting clean and maintainable code.
"""
import os
import shutil
import json
from PIL import Image
import torch
from torchvision import transforms
from modules.nets.model_factory import model_factory

classes_dict = {0: "FrontalAP",
                1: "FrontalPA",
                2: "Lateral"}

def predicts_ims_from_dir(images_path: str,
                          checkpoint: str,
                          sort_files: bool,
                          **kwargs):
    """
    Processes images from the specified directory, predicts their classes using the provided model checkpoint,
    and optionally sorts them by copying them into subdirectories based on their classes.

    Parameters:
        images_path (str): Directory containing the images to predict.
        checkpoint (str): Path to the model checkpoint.
        sort_files (bool): Whether to sort images into class-based subdirectories (by copying).

    Saves:
        A JSON file in the specified directory or in a new 'sorted_files' directory containing the filenames and their predicted classes.
    This function is usually invoked by `main.py` with command-line arguments to automate the image prediction and sorting workflow.
    """
    checkpoint_dict = torch.load(checkpoint)
    training_init_params = checkpoint_dict["training_init_params"]
    model_init_params = training_init_params["model_init_params"]
    im_size = model_init_params["im_size"]
    triplicate_channel = training_init_params['training_hyperparams']['triplicate_channel']
    model = model_factory(checkpoint=checkpoint_dict)
    model.eval()

    if isinstance(im_size, list):
        im_size = tuple(im_size)
    comp_transforms = transforms.Compose([transforms.Resize(im_size), 
                                          transforms.Grayscale(), 
                                          transforms.ToTensor()])

    outcome_dict = {}
    with torch.no_grad():
        for file in os.listdir(images_path):
            fpath = os.path.join(images_path, file)
            if os.path.isfile(fpath):
                image = Image.open(fpath)
                image = comp_transforms(image)
                if triplicate_channel:
                    image = image.repeat(3, 1, 1)
                image = image.unsqueeze(0)
                pred = model(image)
                _, quantized_pred = torch.max(pred, 1)
                pred_class = quantized_pred.item()
                outcome_dict[file] = classes_dict[pred_class]

    if sort_files:
        classes_paths = {}
        sorted_files_path = os.path.join(images_path, "sorted_files")
        os.makedirs(sorted_files_path, exist_ok=True)

        for class_name in classes_dict.values():
            class_path = os.path.join(sorted_files_path, class_name)
            classes_paths[class_name] = class_path
            os.makedirs(class_path, exist_ok=True)
        
        for file, pred_class in outcome_dict.items():
            source_path = os.path.join(images_path, file)
            class_path = classes_paths[pred_class]
            destination_path = os.path.join(class_path, file)
            shutil.copy2(source_path, destination_path)
        print(f"Finished copying and sorting files in directory: {sorted_files_path}")
        outcome_json_path = os.path.join(sorted_files_path, "image_classifiaction_data.json")
    else:
        outcome_json_path = os.path.join(images_path, "image_classifiaction_data.json")

    with open(outcome_json_path, 'w') as outcome_file:
        json.dump(outcome_dict, outcome_file, indent=4)
    print(f"Classification data JSON saved at: {outcome_json_path}")