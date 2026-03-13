import os
import random
import shutil

# original dataset path
dataset_dir = "dataset"

# output directory
output_dir = "dataset_split"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = os.listdir(dataset_dir)

for cls in classes:

    class_path = os.path.join(dataset_dir, cls)

    if not os.path.isdir(class_path):
        continue

    # Handle the case where dataset has a nested folder with the same class name (e.g. dataset/0/0)
    nested_path = os.path.join(class_path, cls)
    if os.path.isdir(nested_path):
        class_path = nested_path

    images = os.listdir(class_path)
    # Filter out any non-files just in case
    images = [f for f in images if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(images)

    total = len(images)

    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split_name, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):

        split_dir = os.path.join(output_dir, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in split_imgs:

            src = os.path.join(class_path, img)
            dst = os.path.join(split_dir, img)

            shutil.copy(src, dst)

print("Dataset split complete!")