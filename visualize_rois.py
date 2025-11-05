import os
import random
import matplotlib.pyplot as plt
import cv2

root = 'defect_rois_organized'
classes = os.listdir(root)
plt.figure(figsize=(15, 8))
for i, c in enumerate(classes):
    files = os.listdir(os.path.join(root, c))
    sample = random.choice(files)
    img = cv2.imread(os.path.join(root, c, sample))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, len(classes)//2 + 1, i + 1)
    plt.imshow(img)
    plt.title(c)
    plt.axis("off")
plt.tight_layout()
plt.show()
