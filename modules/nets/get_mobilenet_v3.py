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

# Mapping between model size strings and their corresponding pretrained weight objects.
weights_map = {'mobilenet_v3_small': MobileNet_V3_Small_Weights.DEFAULT,
               'mobilenet_v3_large': MobileNet_V3_Large_Weights.DEFAULT}

# Mapping between model size strings and their corresponding function calls in torchvision.models.
model_types_map = {'mobilenet_v3_small': models.mobilenet_v3_small,
                   'mobilenet_v3_large': models.mobilenet_v3_large}


def get_model(model_type: str = "mobilenet_v3_small",
              in_channels: int = 1,
              adapt_pretrained_weights_for_inchannels: bool = True,
              num_classes: int = 3,
              freeze_first_clayer: bool = True,
              freeze_middle_layers: bool = True,
              freeze_classifier: bool = False,
              freeze_class_out_layer: bool = False,
              **kwargs):
    """
    Initializes and returns a MobileNet v3 model (either small or large) with custom configurations.
    
    This function allows for significant customization of the MobileNet v3 model, accommodating a variety of use cases.
    Customization options include selecting the model size (small or large), setting the number of input channels and adapting 
    pretrained weights for these channels, specifying the number of output classes, and selectively freezing components of the model.
    
    Params:
        model_type (str): Specifies the model variant to initialize ('mobilenet_v3_small' or 'mobilenet_v3_large'). Default is 'mobilenet_v3_small'.
        in_channels (int): The number of input channels for the model. Default is 1.
        adapt_pretrained_weights_for_inchannels (bool): Whether to adapt pretrained weights for the specified number of input channels.
                                                        The weights for each new channels will be set to mean of the pretrained weights if True. Default is True.
        num_classes (int): The number of output classes for the model's classifier. Default is 3.
        freeze_first_clayer (bool): If True, the first convolutional layer of the model will be frozen (weights will not be updated during training). Default is False.
        freeze_middle_layers (bool): If True, all middle layers of the model will be frozen. Default is True.
        freeze_classifier (bool): If True, the classifier layers, excluding the last layer, will be frozen. Default is False.
        freeze_class_out_layer (bool): If True, the output layer of the classifier will be frozen. This parameter inversely affects the freezing of the classifier's output layer. Default is False.

    Returns:
        A PyTorch model (nn.Module) instance of the specified MobileNet version with the applied customizations.
        
    Example:
        To initialize a 'Large' MobileNet v3 model adapted for 3 input channels with pretrained weights, 10 output classes, and selective layer freezing, use:
            model = get_model(model_type='Large', in_channels=3, num_classes=10, freeze_middle_layers=True, freeze_classifier=True, freeze_class_out_layer=False)
    """
    
    model_type = model_type.lower()

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
    if freeze_first_clayer:
        for parameter in model.features[0][0].parameters():
            parameter.requires_grad = False

    if freeze_middle_layers:
        for layer_index in range(1, len(model.features)):
            for parameter in model.features[layer_index].parameters():
                parameter.requires_grad = False

    if freeze_classifier:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = False

    # Unfreezing the classifier's output layer if specified.
    if not freeze_class_out_layer:
        for parameter in model.classifier[-1].parameters():
            parameter.requires_grad = True
    
    return model
