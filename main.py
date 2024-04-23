import argparse
from modules.orchestrators import *

supported_models = ["chex_conv_classifier", "mobilenet_v3_small", "mobilenet_v3_large"]
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(dest="mode", required=True, description="Mode selection. Possible choices are ['train', 'test']")

parser_train = subparsers.add_parser("train", help="Model training mode.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_train.add_argument('model_type', type=str, choices=["chex_conv_classifier", "mobilenet_v3_small", "mobilenet_v3_large"], help="The type ot the model to train.")
parser_train.add_argument('dataset_root', type=str, help="Path to the dataset root directory.")
parser_train.add_argument('checkpoints_root', type=str, help="Path to checkpoints root for saving new checkpoints")
parser_train.add_argument('-m', '--model_config', type=str, help="Path to the chosen model's config JSON. If not provided, the model will be intialized with the default parameters, broadly desribed in README.md")
parser_train.add_argument('-o', '--optim_config', type=str, default="./configs/optimizer_config.json", help="Path to the optimizer's config JSON")
parser_train.add_argument('-p', '--hyperparams', type=str, default="./configs/hyperparams_config.json", help="Path to the hyperparameters config JSON for training.")
parser_train.add_argument('-f', '--filenames_config', type=str, default='./configs/filenames_config.json', help="Path to the filenames config JSON.")
parser_train.add_argument('-t', '--triplicate_channel', action='store_true', help="Enables triplicating the input image tensor. Use if the model is configured to require three input channels instead of one.")

parser_test = subparsers.add_parser("test", help="Model testing mode.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_test.add_argument('dataset_root', type=str, help="Path to the dataset root directory.")
parser_test.add_argument('checkpoint', type=str, help="Path to the checkpoint of a tested model (Model type is specified in the checkpoint.)")
parser_test.add_argument('-f', '--filenames_config', type=str, default='./configs/filenames_config.json', help="Path to the filenames config JSON.")
parser_test.add_argument('-b', '--batch_size', type=int, default=64, help="Test batch size.")

parser_predict = subparsers.add_parser("predict", help="Predict classes of images from a given directory and optionally sort the images copying them into subdirectories based on these predicted classes.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_predict.add_argument("images_path", type=str, help="Path to directory containing images.")
parser_predict.add_argument('checkpoint', type=str, help="Path to the checkpoint of a model used to predict images classes. (Model type is specified in the checkpoint.)")
parser_predict.add_argument('-s', '--sort_files', action='store_true', help="Enables sorting the files by copying them into class-named folders inside of 'sorted_files' folder in the main images directory.")

args = parser.parse_args()
args_dict = vars(args)
if args.mode == "train":
    model_train(**args_dict)
elif args.mode == "test":
    model_test(**args_dict)
elif args.mode == "predict":
    predicts_ims_from_dir(**args_dict)
