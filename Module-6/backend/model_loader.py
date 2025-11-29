import torch
import timm

def load_model(model_path="best_efficientnet_b4.pth"):
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
