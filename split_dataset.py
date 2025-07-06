import os
import shutil
import random

def split_dataset(source_dir, trainval_dir, test_dir, test_split=0.1):
    if not os.path.exists(trainval_dir):
        os.makedirs(trainval_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_index = int(len(images) * (1 - test_split))
        trainval_images = images[:split_index]
        test_images = images[split_index:]

        # Create class subfolders
        trainval_class_path = os.path.join(trainval_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)
        os.makedirs(trainval_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        # Move files
        for img in trainval_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(trainval_class_path, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_path, img))

    print("✅ Dataset split completed.")

# Example usage:
source = r"C:\Users\Gayatri C B\Desktop\LF_disease\t"
trainval = r"C:\Users\Gayatri C B\Desktop\LF_disease\trdataseain_val"
test = r"C:\Users\Gayatri C B\Desktop\LF_disease\test"

split_dataset(source, trainval, test, test_split=0.1)
