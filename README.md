## 1. Downloading CheXpert dataset and preparing its file structure

- Download the dataset from the official website. [LINK](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)
- Unzip all the files listed below:
	- `CheXpert-v1.0 batch 1 (validate & csv).zip`
	- `CheXpert-v1.0 batch 2 (train 1).zip`
	- `CheXpert-v1.0 batch 3 (train 2).zip`
	- `CheXpert-v1.0 batch 4 (train 3).zip`
	
- Move all the `patientXXXXX` folders (XXXXX stands for the patient number) from the `CheXpert-v1.0 batch 1 (validate & csv)/valid` directory into the new `valid_data` folder within the CheXpert root directory.

- Move both the `train.csv` and `valid.csv` files from the `CheXpert-v1.0 batch 1 (validate & csv)` to the CheXpert root directory

- Remove the empty `CheXpert-v1.0 batch 1 (validate & csv)` folder

- Move all the `patientXXXXX` folders (XXXXX stands for the patient number) from all the unzipped `CheXpert-v1.0 batch x (train y)` folders into the new `train_data` folder within the CheXpert root directory.

- Remove all the empty `CheXpert-v1.0 batch x (train y)` folders.
- Remove all the other unnecessary files not mentioned above (mostly CSVs and other models stats files). Leave the `README.md` if you want. NOTE: Removing the zip files mentioned above is not recommended! If you mess anything with the file structure, fixing it would involve downloading the whole dataset again. Keep the zip files to save the time in case of the file moving mistake.

- At this stage, the file structure should be as follows::
	- `CheXpert_root_directory` (you can name it as you wish)
		- `train_data`
			- `patient00001`
			- `patient00002`
			- `...`
			- `patient64540`
		- `valid_data`
			- `patient6541`
			- `patient6542`
			- `...`
			- `patient64740`
		- `train.csv` (file containing all the training images information and the labels)
		- `valid.csv` (file containing all the validation images information and the labels)
		- `CheXpert-v1.0 batch 1 (validate & csv).zip`
		- `CheXpert-v1.0 batch 2 (train 1).zip`
		- `CheXpert-v1.0 batch 3 (train 2).zip`
		- `CheXpert-v1.0 batch 4 (train 3).zip`
		- `README.md`(optional)


## 2. CSV Modification Tool

I have developed the "csv_modifier.py" tool, which modifies the original CSV files (in this case, "train.csv" and "valid.csv") from the CheXpert dataset. The tool prepares the "Path" column for further use, concatenates the "Frontal/Lateral" and "AP/PA" columns, and drops unnecessary columns. At the end, it creates the new csv file with the desired name.

## 3. custom_CheXpert_Dataset.py Module

I have created the "custom_CheXpert_Dataset.py" module, defining the CheXpert class. This class inherits from the PyTorch Dataset class and returns a tuple with a CheXpert image and its ground truth label (integer encoding). Custom transforms can be applied to the image, and both resizing (to 320 x 320 pixels) and converting the image to grayscale are supported.

## 4. Classifier Module (classifier.py)

The "classifier.py" module defines a custom convolutional neural network classifier. It includes three convolutional layers followed by a fully connected layer for image classification. The hyperparameters for these layers, as well as the input image size, are configurable. This module plays a crucial role in the project by providing the architecture for training and evaluating model on the CheXpert dataset.

## 5. Model Training and Evaluation in main.py

In the "main.py" file, I have implemented the main script for the project, where model training and evaluation take place. This script leverages the CheXpert dataset loaded using the "custom_CheXpert_Dataset.py" module and initializes the custom convolutional neural network classifier defined in the "classifier.py" module. The hyperparameters for the network and dataset are configured in this script, providing a solid foundation for model training and evaluation. The key functionalities added to "main.py" include:

- Loading and preprocessing the CheXpert dataset.
- Defining the neural network architecture using the custom classifier.
- Configuring hyperparameters such as learning rate, momentum, and batch size.
- Training the model using a training loop.
- Saving checkpoints of the model's state and optimizer's state after each epoch.
- Evaluating the model on validation and test datasets.

This script serves as the central component of the project, orchestrating the entire machine learning pipeline.