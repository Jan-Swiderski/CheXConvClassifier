# Project Progress

## 1. Moving CheXpert Files

All CheXpert files from "CheXpert-v1.0 batch x (train y)" folders have been consolidated into a single "train_data" folder within the CheXpert root directory.

## 2. CSV Modification Tool

I have developed the "csv_modifier.py" tool, which modifies the original CSV files (in this case, "train.csv") from the CheXpert dataset. The tool prepares the "Path" column for further use, concatenates the "Frontal/Lateral" and "AP/PA" columns, and drops unnecessary columns. At the end, it creates the new "train_data.csv" file.

## 3. custom_CheXpert_Dataset.py Module

I have created the "custom_CheXpert_Dataset.py" module, defining the CheXpert class. This class inherits from the PyTorch Dataset class and returns a tuple with a CheXpert image and its ground truth label. Specific transforms are applied to the image, and resizing to 320 x 320 pixels is also supported.
