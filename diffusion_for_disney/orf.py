import os
import shutil

base_dir = os.path.join("data", "cartoon")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
output_dir = os.path.join(base_dir, "images")

os.makedirs(output_dir, exist_ok=True)

image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp")

def move_images_from_dir(input_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                old_path = os.path.join(root, file)
                new_file_name = f"{os.path.basename(root)}_{file}"
                new_path = os.path.join(output_dir, new_file_name)

                counter = 1
                while os.path.exists(new_path):
                    new_file_name = f"{os.path.basename(root)}_{counter}_{file}"
                    new_path = os.path.join(output_dir, new_file_name)
                    counter += 1

                shutil.move(old_path, new_path)
                print(f"Moved: {old_path} -> {new_path}")

move_images_from_dir(train_dir)
move_images_from_dir(test_dir)
