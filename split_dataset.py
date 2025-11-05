import os, shutil, random

source_dir = "dataset/test_images"
train_dir = "dataset/train"
test_dir = "dataset/test"

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each defect category folder
for defect in os.listdir(source_dir):
    defect_path = os.path.join(source_dir, defect)
    if not os.path.isdir(defect_path):
        continue

    images = [f for f in os.listdir(defect_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    split_index = int(0.8 * len(images))

    train_images = images[:split_index]
    test_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, defect), exist_ok=True)
    os.makedirs(os.path.join(test_dir, defect), exist_ok=True)

    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(defect_path, img), os.path.join(train_dir, defect, img))
    for img in test_images:
        shutil.copy(os.path.join(defect_path, img), os.path.join(test_dir, defect, img))

    print(f"✅ {defect}: {len(train_images)} train, {len(test_images)} test")

print("\nDataset split complete!")
