"""Model factory: builds a pretrained backbone (torchvision or timm) with its
classification head resized to num_classes, plus freeze/unfreeze helpers for
staged fine-tuning (see classifier/config.py's FREEZE options).
"""

from torchvision.models import efficientnet_b0, squeezenet1_1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, resnet18
from torch import nn
import timm

def build_model(model_name, num_classes, freeze):
    """Build `model_name` with a fresh head sized to num_classes.

    freeze="none" leaves every parameter trainable; any other value freezes
    the backbone and leaves only the head trainable (see get_backbone/unfreeze
    for staged fine-tuning after this initial freeze).
    """
    if model_name == "efficientnet_b0":
        return efficient_net_b0(num_classes, freeze)
    elif model_name == "mobilenet_v2":
        return mobile_net_v2(num_classes, freeze)
    elif model_name == "mobilenet_v3":
        return mobile_net_v3(num_classes, freeze)
    elif model_name == "squeezenet":
        return squeeze_net(num_classes, freeze)
    elif model_name == "resnet":
        return res_net(num_classes, freeze)
    elif model_name == "efficientnet_lite0":
        return efficientnet_lite0(num_classes, freeze)
    elif model_name == "efficientnet_lite1":
        return efficientnet_lite1(num_classes, freeze)
    elif model_name == "ghostnet":
        return ghost_net(num_classes, freeze)
    elif model_name == "mobilenet_v3_small":
        return mobile_net_v3_small(num_classes, freeze)

    raise ValueError(f"Unknown model_name='{model_name}'")

def get_backbone(model, model_name):
    """Return the backbone (feature extractor) module for a given model."""
    if model_name in ("mobilenet_v3", "mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0"):
        return model.features
    elif model_name == "resnet":
        return nn.Sequential(model.layer1, model.layer2, model.layer3, model.layer4)
    elif model_name == "squeezenet":
        return model.features
    elif model_name in ("efficientnet_lite0", "efficientnet_lite1", "ghostnet"):
        return model.blocks
    else:
        raise ValueError(f"Unknown model_name='{model_name}'")


def freeze_all(model):
    """Set requires_grad=False on every parameter in model."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(module):
    """Set requires_grad=True on every parameter in module."""
    for p in module.parameters():
        p.requires_grad = True

# ---- Torchvision models ----
# Each loads ImageNet-pretrained weights, swaps the final classifier layer for one
# sized to num_classes, and (if freeze != "none") freezes everything except that
# new head.

def efficient_net_b0(num_classes, freeze):
    model = efficientnet_b0(weights = 'IMAGENET1K_V1', progress = True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier[1])

    return model

def mobile_net_v2(num_classes, freeze):
    model = mobilenet_v2(weights = 'IMAGENET1K_V1', progress = True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier[1])

    return model

def mobile_net_v3(num_classes, freeze):
    # v3-large (see mobile_net_v3_small for the small variant)
    model = mobilenet_v3_large(weights = 'IMAGENET1K_V1', progress = True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier[3])

    return model

def mobile_net_v3_small(num_classes, freeze):
    model = mobilenet_v3_small(weights='IMAGENET1K_V1', progress=True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier[3])

    return model

def squeeze_net(num_classes, freeze):
    # Use SqueezeNet 1.1 over SqueezeNet 1.0 as better
    model = squeezenet1_1(weights = 'IMAGENET1K_V1', progress = True)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier[1])

    return model

def res_net(num_classes, freeze):
    # Use resnet18 initially
    model = resnet18(weights = 'IMAGENET1K_V1', progress = True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.fc)

    return model

# ---- TIMM models ----
# Same pattern as above, via timm's built-in num_classes= head replacement.

def efficientnet_lite0(num_classes, freeze):
    model = timm.create_model("efficientnet_lite0", pretrained=True, num_classes=num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier)

    return model

def efficientnet_lite1(num_classes, freeze):
    # ! timm has no pretrained checkpoint for this variant — pretrained=True is a no-op here.
    model = timm.create_model("efficientnet_lite1", pretrained=True, num_classes=num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier)

    return model

def ghost_net(num_classes, freeze):
    model = timm.create_model("ghostnet_100", pretrained=True, num_classes=num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier)

    return model