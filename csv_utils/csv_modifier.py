import pandas as pd
import os
from dotenv import load_dotenv

# Hyperparameter - decide whether to modify train.csv or valid.csv. If true, valid.csv is selected.
if_valid = True

# Load environment variables
load_dotenv()

source_csv_name = 'valid.csv'
out_csv_name = 'valid_data_info.csv'

# Get the path to CheXpert dataset root directory
chexpert_root = os.getenv('PATH_TO_CHEXPERT_ROOT')

# Construct the path to the CSV file containing training data information (csv file)
csv_path = os.path.join(chexpert_root, 'CheXpert-v1.0 batch 1 (validate & csv)', source_csv_name)

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_path)

if if_valid:
    # Remove the 'CheXpert-v1.0/valid/' prefix from the 'Path' column
    df['Path'] = df['Path'].str.replace('CheXpert-v1.0/valid/', '')
else:
    # Remove the 'CheXpert-v1.0/train/' prefix from the 'Path' column
    df['Path'] = df['Path'].str.replace('CheXpert-v1.0/train/', '')

# Select columns from 'Path' to 'AP/PA' (inclusive)
df = df.loc[:, 'Path':'AP/PA']

# Combine 'Frontal/Lateral' and 'AP/PA' columns into a new 'Frontal(AP/PA)/Lateral' column
df['Frontal(AP/PA)/Lateral'] = df['Frontal/Lateral'].str.cat(df['AP/PA'].fillna(''), sep='')

# Drop the 'Frontal/Lateral' and 'AP/PA' columns
df = df.drop(columns=['Frontal/Lateral', 'AP/PA'])

# Save the modified DataFrame to a new CSV file in the CheXpert root directory
df.to_csv(os.path.join(chexpert_root, out_csv_name), index=False)