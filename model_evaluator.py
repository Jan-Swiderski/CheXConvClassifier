import os
from dotenv import load_dotenv
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from modules.dataset.custom_chexpert_dataset import CheXpert
from modules.evaluation.model_eval import model_eval

if __name__ == "__main__":
    load_dotenv(dotenv_path = "./.env", override = True)
    CHEXPERT_ROOT = os.getenv('PATH_TO_CHEXPERT_ROOT')
    CHECKPOINTS_ROOT = os.getenv('PATH_TO_CHECKPOINTS_DIR')

    PATH_TO_CHECKPOINT = "2024-02-24_23-08_mobile_net_v3_large/checkpoint_acc98-64epoch006.pth"

    CHECKPOINT_DIR = os.path.join(CHECKPOINTS_ROOT, PATH_TO_CHECKPOINT)

    TEST_DINFO_FILENAME = "medsize_test_data_info.csv"
    TEST_IMAGES_DIRNAME = "train_data"

    TEST_BATCH_SIZE = 64  
    IM_SIZE = (128, 128)

    WEIGHTS = MobileNet_V3_Large_Weights.DEFAULT
    mobilenet_v3_large = models.mobilenet_v3_large(weights=WEIGHTS)

    # Retrieve the first convolutional layer
    first_conv_layer = mobilenet_v3_large.features[0][0]

    # Create a new first convolutional layer that accepts 1 input channel
    new_first_conv_layer = torch.nn.Conv2d(in_channels=1,
                                            out_channels=first_conv_layer.out_channels,
                                            kernel_size=first_conv_layer.kernel_size,
                                            stride=first_conv_layer.stride,
                                            padding=first_conv_layer.padding,
                                            bias=first_conv_layer.bias)

    # Replace the original first layer with the new one
    mobilenet_v3_large.features[0][0] = new_first_conv_layer

    # Adjust the last layer for the new number of classes
    last_layer_in_features = mobilenet_v3_large.classifier[-1].in_features
    mobilenet_v3_large.classifier[-1] = torch.nn.Linear(last_layer_in_features, 3)

    checkpoint = torch.load(os.path.join(CHECKPOINTS_ROOT, PATH_TO_CHECKPOINT))

    mobilenet_v3_large.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = CheXpert(root_dir=CHEXPERT_ROOT,
                            dinfo_filename=TEST_DINFO_FILENAME,
                            images_dirname=TEST_IMAGES_DIRNAME,
                            custom_size=IM_SIZE,
                            custom_transforms=transforms.ToTensor())
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=TEST_BATCH_SIZE,
                             shuffle=False)
    
    criterion = nn.CrossEntropyLoss()


    total_loss, av_loss, accuracy = model_eval(model=mobilenet_v3_large,
                                    dataloader=test_loader,
                                    criterion=criterion)
    
    print(f"Test_accuracy: {accuracy}%")