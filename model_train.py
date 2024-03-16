import os
from dotenv import load_dotenv
from modules.create_checkpoints_subdir import create_checkpoints_subdir


if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv(dotenv_path = "./.env", override = True)
    CHEXPERT_ROOT = os.getenv('PATH_TO_CHEXPERT_ROOT')
    CHECKPOINTS_ROOT = os.getenv('PATH_TO_CHECKPOINTS_DIR')
        
    # Define important filenames.
    TRAIN_DINFO_FILENAME = 'small_train_data_info.csv'
    TRAIN_IMAGES_DIRNAME = 'train_data'

    VALID_DINFO_FILENAME = 'small_valid_data_info.csv'
    VALID_IMAGES_DIRNAME = TRAIN_IMAGES_DIRNAME

    TEST_DINFO_FILENAME = "small_test_data_info.csv"
    TEST_IMAGES_DIRNAME = TRAIN_IMAGES_DIRNAME

    # Hyperparameters
    
    OPTIMIZER_MODULE_NAME = "torch.optim"
    OPTIMIZER_CLASS_NAME = "SGD"
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9

    optimizer_type_info = {'optimizer_module_name': OPTIMIZER_MODULE_NAME,
                           'optimizer_class_name': OPTIMIZER_CLASS_NAME,}
    
    optimizer_init_params = {'optimizer_type_info': optimizer_type_info,
                             'lr': LEARNING_RATE,
                             'momentum': MOMENTUM}

    MAX_EPOCHS = 50
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    PATIENCE = 5 # Number of epochs with no improvement after which training will be stopped.
    MIN_ACC_IMPROVEMENT = 0.5 # Minimum improvement in validation accuracy to qualify as an improvement.
    MIN_MEM_AV_MB = 1024

    training_hyperparams = {'train_batch_size': TRAIN_BATCH_SIZE,
                            'valid_batch_size': VALID_BATCH_SIZE,
                            'test_batch_size': TEST_BATCH_SIZE,
                            'patience': PATIENCE,
                            'min_acc_improvement': MIN_ACC_IMPROVEMENT,
                            'min_mem_av_mb': MIN_MEM_AV_MB}


    MODEL_TYPE = 'classifier'

    L1_OUT_CHANN = 8 # Number of filters in the first convolutional layer
    L1_KERNEL_SIZE = 5 # Size of the first layer's kernel
    L1_STRIDE = 1 # Stride of the first layer
    L2_OUT_CHANN = 16 # Number of filters in the second convolutional layer
    L2_KERNEL_SIZE = 3 # Size of the second layer's kernel
    L2_STRIDE = 1 # Stride of the second layer
    L3_OUT_CHANN = 32  # Number of channels in the third convolutional layer
    L3_KERNEL_SIZE = 5 # Size of the thrid layer's kernel
    L3_STRIDE = 1 # Stride of the third layer
    FC_OUT_FEATURES = 32 #Output features of the first fully connected layer
    IM_SIZE = (128, 128) # Input image size
    
   
    model_init_params = {'model_type': MODEL_TYPE,
                        'l1_kernel_size': L1_KERNEL_SIZE,
                        'l1_stride': L1_STRIDE,
                        'l1_out_chann': L1_OUT_CHANN,
                        'l2_kernel_size': L2_KERNEL_SIZE,
                        'l2_stride': L2_STRIDE,
                        'l2_out_chann': L2_OUT_CHANN,
                        'l3_kernel_size': L3_KERNEL_SIZE,
                        'l3_stride': L3_STRIDE,
                        'l3_out_chann': L3_OUT_CHANN,
                        'fc_out_features': FC_OUT_FEATURES,
                        'im_size': IM_SIZE}                      
                       
    training_init_params = {'training_hyperparams': training_hyperparams,
                            'model_init_params': model_init_params,
                            'optimizer_init_params': optimizer_init_params}
    
    create_checkpoints_subdir(checkpoints_root=CHECKPOINTS_ROOT,
                              model_type=MODEL_TYPE)