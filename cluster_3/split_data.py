import os
import shutil
import random
import math
from pathlib import Path

TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VAL_RATIO = 0.1

SRC_IMAGES_DIR = Path("images")
SRC_LABELS_DIR = Path("labels")

DEST_ROOT = Path("dataset")

def setup_directories():
    subsets = ['train', 'test', 'val']
    for subset in subsets:
        (DEST_ROOT / subset / "images").mkdir(parents=True, exist_ok=True)
        (DEST_ROOT / subset / "labels").mkdir(parents=True, exist_ok=True)

def move_files(file_list, subset_name):
    print(f"Moving {len(file_list)} files to {subset_name}...")
    
    for image_file in file_list:
        src_img_path = SRC_IMAGES_DIR / image_file
        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_lbl_path = SRC_LABELS_DIR / label_file
        
        dst_img_path = DEST_ROOT / subset_name / "images" / image_file
        dst_lbl_path = DEST_ROOT / subset_name / "labels" / label_file
        
        if src_img_path.exists():
            shutil.move(str(src_img_path), str(dst_img_path))
            
        if src_lbl_path.exists():
            shutil.move(str(src_lbl_path), str(dst_lbl_path))
        else:
            print(f"Warning: Label missing for {image_file}")

def main():
    if not SRC_IMAGES_DIR.exists():
        print(f"Error: Could not find '{SRC_IMAGES_DIR}'.")
        return

    all_images = [f.name for f in SRC_IMAGES_DIR.iterdir() if f.suffix.lower() == ".jpg"]
    
    total_images = len(all_images)
    if total_images == 0:
        print("No .jpg images found to split.")
        return

    print(f"Found {total_images} images. Shuffling and splitting...")
    
    random.shuffle(all_images)
    
    train_count = math.ceil(total_images * TRAIN_RATIO)
    test_count = math.ceil(total_images * TEST_RATIO)
    
    train_files = all_images[:train_count]
    test_files = all_images[train_count : train_count + test_count]
    val_files = all_images[train_count + test_count :]
    
    setup_directories()
    
    move_files(train_files, 'train')
    move_files(test_files, 'test')
    move_files(val_files, 'val')
    
    if not any(SRC_IMAGES_DIR.iterdir()):
        SRC_IMAGES_DIR.rmdir()
        print(f"Removed empty folder: {SRC_IMAGES_DIR}")
        
    if not any(SRC_LABELS_DIR.iterdir()):
        SRC_LABELS_DIR.rmdir()
        print(f"Removed empty folder: {SRC_LABELS_DIR}")

    print("\nSuccess! Data split into 'dataset/' directory.")
    print(f"Train: {len(train_files)}")
    print(f"Test:  {len(test_files)}")
    print(f"Val:   {len(val_files)}")

if __name__ == "__main__":
    main()