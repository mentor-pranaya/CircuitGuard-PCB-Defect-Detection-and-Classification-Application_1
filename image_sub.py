#image processing and image substraction#


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class PCBDefectPreprocessor:
    def __init__(self, root_dir):
        self.template_dir = os.path.join(root_dir, 'PCB_USED')
        self.test_dir = os.path.join(root_dir, 'images')
        self.output_dir = os.path.join(root_dir, 'PROCESSED_MASKS')
        os.makedirs(self.output_dir, exist_ok=True)
        self.templates = ['01','04','05','06','07','08','09','10','11','12']

def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (512, 512))
        blurred = cv2.medianBlur(resized, 3)
        normed = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        return normed

def align_images(self, template, test):
        """Aligns test image to template using frequency-domain phase correlation."""
        shift = cv2.phaseCorrelate(np.float32(template), np.float32(test))
        dx, dy = shift[0]
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(test, transform, (template.shape[1], template.shape[0]))
        return aligned

def create_mask(self, template, aligned_test):
     
        diff = cv2.absdiff(template, aligned_test)
        denoised = cv2.GaussianBlur(diff, (5,5), 0)
        mask = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 4
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cleaned


def process_all(self):
        defect_classes = [
            'Missing_hole', 'Mouse_bite', 'Open_circuit',
            'Short', 'Spur', 'Spurious_copper'
        ]

        for defect in defect_classes:
            print(f"\nProcessing: {defect}")
            in_dir = os.path.join(self.test_dir, defect)
            out_dir = os.path.join(self.output_dir, defect)
            os.makedirs(out_dir, exist_ok=True)

            for file in os.listdir(in_dir):
                if not file.lower().endswith('.jpg'):
                    continue
                template_id = file.split('_')[0]
                if template_id not in self.templates:
                    continue

                template_path = os.path.join(self.template_dir, f"{template_id}.JPG")
                test_path = os.path.join(in_dir, file)

                temp_img = cv2.imread(template_path)
                test_img = cv2.imread(test_path)
                if temp_img is None or test_img is None:
                    continue

                pre_t = self.preprocess(temp_img)
                pre_x = self.preprocess(test_img)

                aligned = self.align_images(pre_t, pre_x)
                mask = self.create_mask(pre_t, aligned)

                save_name = f"mask_{file.split('.')[0]}.png"
                cv2.imwrite(os.path.join(out_dir, save_name), mask)

                # Show a few visual samples
                if file.endswith('_01.JPG'):
                    self.visualize(pre_x, aligned, mask, defect, template_id)

        print("\nAll defect masks created successfully.")

def visualize(self, test, aligned, mask, defect, temp_id):
        plt.figure(figsize=(14,4))
        plt.suptitle(f"Defect: {defect} | Template: {temp_id}", fontsize=11)
        plt.subplot(1,3,1); plt.imshow(test, cmap='gray'); plt.title("Original Test")
        plt.subplot(1,3,2); plt.imshow(aligned, cmap='gray'); plt.title("Aligned")
        plt.subplot(1,3,3); plt.imshow(mask, cmap='gray'); plt.title("Defect Mask")
        plt.show()


if __name__ == "__main__":
    data_root = os.path.join('..', 'Data', 'PCB_DATASET')
    pipeline = PCBDefectPreprocessor(data_root)
    pipeline.process_all()
