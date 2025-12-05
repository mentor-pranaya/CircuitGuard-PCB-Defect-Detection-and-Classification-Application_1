import torch
import timm
import torch.nn as nn

def create_model(model_name: str = "efficientnet_b0", num_classes: int = 6, pretrained: bool = True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def load_model_checkpoint(path: str, model_name: str = "efficientnet_b0", num_classes: int = 6, device: str = "cpu"):
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(path, map_location=device)
    # accept either state_dict or direct
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        sd = ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

def predict_patch(model, pil_patch, device: str = "cpu"):
    from torchvision import transforms
    import torch.nn.functional as F
    tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    x = tf(pil_patch).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())

        conf = float(probs[idx])
    return idx, conf
