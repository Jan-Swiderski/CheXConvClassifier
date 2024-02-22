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
