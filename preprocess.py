import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

TARGET_SIZE = (128,128)

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def detect_diffs(template_cv2: np.ndarray, test_cv2: np.ndarray, blur: int = 5, thresh_val: int = 40, min_area: int = 50) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray]:
    # align shapes
    if template_cv2.shape != test_cv2.shape:
        template_cv2 = cv2.resize(template_cv2, (test_cv2.shape[1], test_cv2.shape[0]))
    grayA = cv2.cvtColor(template_cv2, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(test_cv2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(grayA, grayB)
    k = blur if blur % 2 == 1 else blur + 1
    diff = cv2.GaussianBlur(diff, (k,k), 0)
    _, th_otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_user = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    th = cv2.bitwise_or(th_otsu, th_user)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area:
            boxes.append((x,y,w,h))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes, th

def annotate_image(test_cv2, boxes, labels, confidences):
    annotated = test_cv2.copy()
    for (x,y,w,h), lbl, conf in zip(boxes, labels, confidences):
        color = (0,200,0)
        cv2.rectangle(annotated, (x,y), (x+w, y+h), color, 2)
        txt = f"{lbl} {conf:.2f}"
        cv2.putText(annotated, txt, (x, max(16, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return annotated
