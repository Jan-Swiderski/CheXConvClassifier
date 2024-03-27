import argparse
from modules.orchestrators.model_train import model_train

supported_models = ["chex_conv_classifier", "mobilenet_v3_small", "mobilenet_v3_large"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('model_type', type=str, choices=["chex_conv_classifier", "mobilenet_v3_small", "mobilenet_v3_large"], help="Choose the model to train, validate, or test.")

parser.add_argument('mode', type=str, choices=['train', 'validate', 'test'], help="Choose whether to train, validate or test the model.")

parser.add_argument('dataset_root', type=str, help="Path to the dataset root directory.")

parser.add_argument('checkpoint', type=str, help="Path to checkpoints root in 'train' modde or path to the checkpoint file in 'validate' or 'test' mode.")

parser.add_argument('-m', '--model_config', type=str, help="Path to the chosen model's config JSON. If not provided, the model will be intialized with the default parameters, broadly desribed in README.md")

parser.add_argument('-o', '--optim_config', type=str, default="./configs/optimizer_config.json", help="Path to the optimizer's config JSON. Use only in 'train' mode.")

parser.add_argument('-p', '--hyperparams', type=str, default="./configs/hyperparams_config.json", help="Path to the hyperparameters config JSON for training, validation, or testing.")

parser.add_argument('-f', '--filenames_config', type=str, default='./configs/filenames_config.json', help="Path to the filenames config JSON.")

args = parser.parse_args()

args_dict = vars(args)

if args.mode == "train":
    model_train(**args_dict)
