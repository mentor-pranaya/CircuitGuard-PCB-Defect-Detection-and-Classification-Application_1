import os
dataset_path = "dataset/test_images"
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    count = len(os.listdir(class_path))
    print(f"Class: {class_name}, Count: {count}")

