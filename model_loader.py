import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

def load_model(model_path="backend/best_efficientnet_b4.pth"):
    # Create base model
    model = efficientnet_b4(weights=None)

    # Replace classifier for 7 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)

    # Load saved state_dict
    state_dict = torch.load(model_path, map_location="cpu")

    # Remove classifier weights (1000 classes)
    keys_to_remove = []
    for key in state_dict.keys():
        if key.startswith("classifier"):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        print("Removing incompatible layer:", key)
        del state_dict[key]

    # Load remaining weights
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model
