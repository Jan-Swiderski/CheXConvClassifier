"""
This module aligns with the model_factory framework, providing a standardized method for initializing MobileNet v3 models, 
either small or large variants, with a high degree of customization. It employs a consistent function name, `get_model`, 
to facilitate seamless integration and uniformity across the project. This standardization aids in distinguishing models 
through module names rather than varying function names, enhancing modularity and flexibility in model selection and initialization.

The `get_model` function within this module is tasked with initializing a MobileNet v3 model based on a comprehensive set 
of parameters allowing for extensive customization. These parameters include the model type, input channel configuration, 
adaptation of pretrained weights for different input channel counts, selective freezing of model components, and specification 
of the number of output classes. Such detailed customization caters to a wide range of applications and requirements, 
providing the user with the tools to tailor the model to their specific needs.

By adhering to this convention, the module not only simplifies the model initialization process but also enhances compatibility 
and interoperability within the model_factory ecosystem, ensuring that models are easily selectable and configurable based on project requirements.
"""
from torch import nn
from torchvision import models
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

# Defining keys required in the initialization dictionary for model customization.
init_required_keys = ['model_type', 'in_channels','adapt_pretrained_weights_for_inchannels', 'freeze_dict','num_classes']

# Defining keys required in the freeze configuration dictionary to specify which parts of the model to freeze.
freeze_required_keys = ['first_conv_layer', 'middle_layers', 'classifier', 'classifier_out_layer']

# Mapping between model size strings and their corresponding pretrained weight objects.
weights_map = {'Small': MobileNet_V3_Small_Weights.DEFAULT,
               'Large': MobileNet_V3_Large_Weights.DEFAULT}

# Mapping between model size strings and their corresponding function calls in torchvision.models.
model_types_map = {'Small': models.mobilenet_v3_small,
                   'Large': models.mobilenet_v3_large}


def get_model(model_init_params: dict):
    """
    Initializes and returns a MobileNet model (v3 small or large) with custom configurations.
    
    The function allows customization such as setting the number of input channels, adapting pretrained
    weights for a different number of input channels than the default, freezing specified parts of the model,
    and setting the number of output classes for the model's classifier.

    Params:
    - model_init_params (dict): A dictionary containing model initialization parameters. The expected keys and their
      meanings are as follows:
        - 'model_type': A string specifying the model type, either 'Small' or 'Large'.
        - 'in_channels': An integer specifying the number of input channels for the model.
        - 'adapt_pretrained_weights_for_inchannels': A boolean indicating whether to adapt pretrained weights
          for the specified number of input channels as the mean of the pretrained first layers's weights.
        - 'freeze_dict': A dictionary specifying which parts of the model to freeze. It should contain the following keys:
                        - 'first_conv_layer': Boolean indicating whether to freeze the first convolutional layer.
                        - 'middle_layers': Boolean indicating whether to freeze the middle layers of the model.
                        - 'classifier': Boolean indicating whether to freeze the classifier layers of the model, excluding the last layer.
                        - 'classifier_out_layer': Boolean indicating whether to freeze the output layer of the classifier.
        - 'num_classes': An integer specifying the number of output classes for the classifier.

    Returns:
    - A PyTorch model (nn.Module) instance of the specified MobileNet version with the applied customizations.
    
    Raises:
    - ValueError: If any required key is missing in `model_init_params` or `freeze_dict`.
    """
    # Checking for missing required keys in the initialization parameters.
    init_missing_keys = [key for key in init_required_keys if key not in model_init_params]
    if init_missing_keys:
        raise ValueError(f"Missing required key(s) in model_init_params: {','.join(init_missing_keys)}")
    
    # Normalizing the model type string to capitalize the first letter (e.g., 'small' -> 'Small').
    model_type = model_init_params['model_type'].capitalize()

    # Extracting other required parameters from the initialization dictionary.
    in_channels = model_init_params['in_channels']
    adapt_pretrained_weights_for_inchannels = model_init_params['adapt_pretrained_weights_for_inchannels']
    freeze_dict = model_init_params['freeze_dict']
    num_classes = model_init_params['num_classes']

    # Checking for missing required keys in the freeze configuration dictionary.
    freeze_missing_keys = [key for key in freeze_required_keys if key not in freeze_dict]

    # Checking for missing required keys in the freeze configuration dictionary.
    if freeze_missing_keys:
        raise ValueError(f"Missing required key(s) in freeze_required_keys: {','.join(freeze_missing_keys)}")


    # Getting the appropriate pretrained weights based on the model type.
    weights = weights_map[model_type]

    # Initializing the model with the specified pretrained weights.
    model = model_types_map[model_type](weights=weights)

    # Accessing the first convolutional layer of the model.
    first_conv_layer = model.features[0][0]

    # If the number of input channels is different from the default, create and replace the first conv layer.
    if in_channels != first_conv_layer.in_channels:

        # Creating a new convolutional layer with the specified number of input channels.
        new_first_conv_layer = nn.Conv2d(in_channels=in_channels,
                            out_channels=first_conv_layer.out_channels,
                            kernel_size=first_conv_layer.kernel_size,
                            stride=first_conv_layer.stride,
                            padding=first_conv_layer.padding,
                            bias=first_conv_layer.bias)
        
        # If adapting pretrained weights for the new number of input channels, adjust the weights accordingly.
        if adapt_pretrained_weights_for_inchannels:

            # Calculating the mean across the channel dimension and repeating it for the specified number of input channels.
            new_weights = first_conv_layer.weight.detach().mean(dim=1, keepdim=True)
            if in_channels != 1:
                new_weights = new_weights.repeat(1, in_channels, 1, 1)

            # Replacing the original first conv layer with the newly created one.
            new_first_conv_layer.weight.data = new_weights


        model.features[0][0] = new_first_conv_layer

    # Replacing the last layer of the classifier to match the specified number of output classes.
    last_layer_in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(last_layer_in_features, num_classes)

    # Freezing the specified parts of the model based on the freeze configuration dictionary.
    if freeze_dict['first_conv_layer']:
        for parameter in model.features[0][0].parameters():
            parameter.requires_grad = False

    if freeze_dict['middle_layers']:
        for layer_index in range(1, len(model.features)):
            for parameter in model.features[layer_index].parameters():
                parameter.requires_grad = False

    if freeze_dict['classifier']:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = False

    # Unfreezing the classifier's output layer if specified.
    if not freeze_dict['classifier_out_layer']:
        for parameter in model.classifier[-1].parameters():
            parameter.requires_grad = True
    
    return model
