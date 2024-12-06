import kagglehub
import shutil
import os
from PIL import Image
from torchvision import transforms

# Download and prepare datasets
# Dataset 1: Industrial Tools Classification
path1 = kagglehub.dataset_download("niharikaamritkar/industrial-tools-classification")
source_path1 = '/root/.cache/kagglehub/datasets/niharikaamritkar/industrial-tools-classification/versions/1'
destination_path1 = '/content/industrial-tools-classification'
shutil.move(source_path1, destination_path1)
print("Dataset 1 Path:", destination_path1)

# Dataset 2: Real-Life Industrial Dataset
path2 = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")
source_path2 = '/root/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2'
destination_path2 = '/content/industrial-tools-classification2'
shutil.move(source_path2, destination_path2)
print("Dataset 2 Path:", destination_path2)

# Image augmentation setup
base_dir = "/content/industrial-tools-classification/train-20240626T051434Z-001/train"
augmented_dir = "/content/industrial-tools-classification/augmented2_train"

augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=20, scale=(0.8, 1.2)),
])

def augment_images_in_directory(input_dir, output_dir, num_augmented_copies=5):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)
        for file_name in files:
            if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file_name)
                img = Image.open(file_path).convert('RGB')
                img.save(os.path.join(target_dir, file_name))
                for i in range(num_augmented_copies):
                    augmented_img = augmentation_transforms(img)
                    augmented_img.save(os.path.join(target_dir, f"{os.path.splitext(file_name)[0]}_aug_{i}.jpg"))

augment_images_in_directory(base_dir, augmented_dir)
print("Augmentation complete.")
