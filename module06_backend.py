
# module06_backend.py
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time, cv2, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn.functional as F, timm
import torchvision.transforms as T

DEFAULT_IMG_SIZE = 224
DEFAULT_SUBTRACT_RESIZE = (1024,1024)
DEFAULT_MIN_AREA = 80
DEFAULT_PAD = 6
DEFAULT_MAX_ROIS = 40

def load_checkpoint(ckpt_path: Path, model_name: str="efficientnet_b0", device: Optional[torch.device]=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "model_state" not in ckpt:
        raise RuntimeError("Checkpoint missing 'model_state'")
    label2idx = ckpt.get("label2idx")
    if label2idx is None:
        raise RuntimeError("Checkpoint missing 'label2idx'")
    idx2label = {int(v): k for k,v in label2idx.items()}
    model = timm.create_model(model_name, pretrained=False, num_classes=len(idx2label))
    model.load_state_dict(ckpt["model_state"])
    model.to(device); model.eval()
    return model, idx2label, device

def preprocess_and_subtract(template_path: Path, test_path: Path, resize_to: Optional[Tuple[int,int]]=DEFAULT_SUBTRACT_RESIZE):
    t = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    s = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)
    if t is None or s is None:
        raise ValueError("Cannot read images")
    if resize_to:
        t = cv2.resize(t, resize_to); s = cv2.resize(s, resize_to)
    diff = cv2.absdiff(s, t)
    diff_blur = cv2.GaussianBlur(diff, (5,5), 0)
    _, th = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return diff, th

def extract_rois_from_mask(test_img_path: Path, mask: np.ndarray, min_area:int=DEFAULT_MIN_AREA, pad:int=DEFAULT_PAD, max_rois:Optional[int]=DEFAULT_MAX_ROIS):
    img_bgr = cv2.imread(str(test_img_path))
    if img_bgr is None:
        raise ValueError("Cannot read test image")
    if mask.shape != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rois=[]
    for i,c in enumerate(contours):
        if max_rois is not None and i>=max_rois: break
        a = cv2.contourArea(c)
        if a < min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(img_bgr.shape[1], x+w+pad); y1 = min(img_bgr.shape[0], y+h+pad)
        crop = img_bgr[y0:y1, x0:x1]
        rois.append(((int(x0),int(y0),int(x1),int(y1)), crop))
    return rois

def crop_to_tensor(crop_bgr: np.ndarray, img_size:int=DEFAULT_IMG_SIZE):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    transform = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return transform(pil).unsqueeze(0)

def predict_rois(model:torch.nn.Module, idx2label:Dict[int,str], rois, device:torch.device, img_size:int=DEFAULT_IMG_SIZE):
    detections=[]
    model.eval()
    for i,(bbox,crop) in enumerate(rois):
        inp = crop_to_tensor(crop, img_size).to(device)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = int(probs.argmax()); score = float(probs[top_idx]); label = idx2label[top_idx]
        detections.append({"roi_idx":i, "bbox":bbox, "label":label, "score":score, "probs":probs})
    return detections

def annotate_image(test_img_path:Path, detections, out_path:Optional[Path]=None):
    img_bgr = cv2.imread(str(test_img_path)); out = img_bgr.copy()
    for d in detections:
        x0,y0,x1,y1 = d["bbox"]
        lab = f"{d['label']} {d['score']:.2f}"
        cv2.rectangle(out, (x0,y0),(x1,y1),(0,0,255),2)
        ((tw,th),_) = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0,y0-th-6),(x0+tw+4,y0),(0,0,255),-1)
        cv2.putText(out, lab, (x0+2,y0-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
    if out_path: cv2.imwrite(str(out_path), out)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def run_pipeline_on_pair(template_path:Path, test_path:Path, ckpt_path:Path, out_dir:Path,
                         resize_to=(800,800), min_area=80, pad=6, max_rois=10, img_size=224, model_name="efficientnet_b0"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, idx2label, device = load_checkpoint(ckpt_path, model_name=model_name, device=None)
    diff, mask = preprocess_and_subtract(template_path, test_path, resize_to=resize_to)
    rois = extract_rois_from_mask(test_path, mask, min_area=min_area, pad=pad, max_rois=max_rois)
    detections = predict_rois(model, idx2label, rois, device, img_size=img_size)
    annotated_path = out_dir / f"annotated_{test_path.stem}.png"
    annotate_image(test_path, detections, out_path=annotated_path)
    rows=[]
    for d in detections:
        x0,y0,x1,y1 = d["bbox"]
        rows.append({"image":test_path.name, "roi_idx":d["roi_idx"], "x0":x0,"y0":y0,"x1":x1,"y1":y1, "label":d["label"], "score":d["score"]})
    csv_path = out_dir / f"detections_{test_path.stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"annotated_path":annotated_path, "csv_path":csv_path, "detections":detections, "elapsed_sec": time.time()-t0}

def run_pipeline_on_folder(images_root:Path, rotation_root:Path, ckpt_path:Path, out_dir:Path, resize_to=(800,800), max_rois=10, img_size=224):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary=[]
    classes = sorted([d.name for d in images_root.iterdir() if d.is_dir()])
    for cls in classes:
        src_dir = images_root / cls
        rot_dir = rotation_root / f"{cls}_rotation"
        if not rot_dir.exists(): continue
        for tpl in sorted([p for p in src_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]):
            test_img = rot_dir / tpl.name
            if not test_img.exists(): continue
            res = run_pipeline_on_pair(tpl, test_img, ckpt_path, out_dir, resize_to=resize_to, max_rois=max_rois, img_size=img_size)
            for d in res["detections"]:
                x0,y0,x1,y1 = d["bbox"]
                summary.append({"test_image": test_img.name, "roi_idx": d["roi_idx"], "x0":x0,"y0":y0,"x1":x1,"y1":y1, "label":d["label"], "score":d["score"], "annotated": str(res["annotated_path"])})
    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "batch_detections_summary.csv", index=False)
    return df
