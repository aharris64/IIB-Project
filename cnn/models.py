from torchvision.models import efficientnet_b0, squeezenet1_1, mobilenet_v2, mobilenet_v3_large, resnet18
from torch import nn
import timm

def build_model(model_name, num_classes, freeze):
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

    return ValueError(f"Unknown model_name='{model_name}'")

# Initially either a full or no freeze (possibly add partial later)

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

# ---- Torchvision Models ---- 

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
    # Use v3 large over v3 small initially
    model = mobilenet_v3_large(weights = 'IMAGENET1K_V1', progress = True)
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

# ---- TIMM ---- 

def efficientnet_lite0(num_classes, freeze):
    model = timm.create_model("efficientnet_lite0", pretrained=True, num_classes=num_classes)

    if freeze != "none":
        freeze_all(model)
        unfreeze(model.classifier)

    return model

def efficientnet_lite1(num_classes, freeze):
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