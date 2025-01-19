from IPython import get_ipython
from IPython.display import display
import os
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")

print("Path to dataset files:", path)

# Define desired folder and move files
desired_folder = "your_desired_folder"
os.makedirs(desired_folder, exist_ok=True)

for filename in os.listdir(path):
    source_path = os.path.join(path, filename)
    destination_path = os.path.join(desired_folder, filename)
    shutil.move(source_path, destination_path)

print(f"Dataset moved to: {desired_folder}")