import os
import shutil
from sklearn.model_selection import train_test_split

# Path to your dataset
dataset_dir = r'D:\New downloads\dataset'

# Create directories for train/validation
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')

# Ensure train/validation directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop through each subdirectory (disease category)
try:
    subdirs = os.listdir(dataset_dir)
    print(f"Found {len(subdirs)} subdirectories")
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        
        # Skip if not a directory or if it's train/validation directory
        if not os.path.isdir(subdir_path) or subdir in ['train', 'validation']:
            continue

        print(f"\nProcessing directory: {subdir}")

        # Create directories for each class in train/validation
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(val_dir, subdir), exist_ok=True)

        # Get all image files in the subdirectory
        images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) < 2:
            print(f"Skipping {subdir}: Not enough images (found {len(images)})")
            continue
            
        print(f"Found {len(images)} images")

        try:
            # Convert images list to numpy array to ensure compatibility
            import numpy as np
            images_array = np.array(images)
            
            # Split the dataset
            train_images, val_images = train_test_split(
                images_array,
                test_size=0.2,
                random_state=42,
                shuffle=True
            )
            
            print(f"Split into {len(train_images)} training and {len(val_images)} validation images")
            
            # Move images to train and validation directories
            for image in train_images:
                src = os.path.join(subdir_path, image)
                dst = os.path.join(train_dir, subdir, image)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"Warning: Source file not found - {src}")
                
            for image in val_images:
                src = os.path.join(subdir_path, image)
                dst = os.path.join(val_dir, subdir, image)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"Warning: Source file not found - {src}")
                
            print(f"Successfully processed {subdir}")
            
        except Exception as e:
            print(f"Error processing split for {subdir}: {str(e)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Dataset directory exists: {os.path.exists(dataset_dir)}")
    print(f"Dataset directory contents: {os.listdir(dataset_dir) if os.path.exists(dataset_dir) else 'N/A'}")
