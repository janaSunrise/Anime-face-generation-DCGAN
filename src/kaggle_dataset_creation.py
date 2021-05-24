# -*- coding: utf-8 -*-
# Define the list of the datasets we'll be downloading
kaggle_datasets = [
  "splcher/animefacedataset",
  "soumikrakshit/anime-faces",
  "scribbless/another-anime-face-dataset"
]

# Get the Kaggle username and API key
import getpass
import os

username = getpass.getpass("Your kaggle username: ")
api_key = getpass.getpass("Your kaggle API Key: ")

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = api_key

# The API command to download a dataset is:
# `kaggle datasets download -d <dataset>`

# Downloading the datasets
import subprocess


def download_kaggle_dataset(dataset):
  result = subprocess.run(["kaggle", "datasets", "download", "-d", dataset], capture_output=True, text=True)
  print("stdout:", result.stdout)
  print("stderr:", result.stderr)


for dataset in kaggle_datasets:
  download_kaggle_dataset(dataset)

# Unzip the datasets
import zipfile


def unzip(dataset):
  filename = dataset.split("/")[1]
  with zipfile.ZipFile(filename + ".zip", 'r') as zip_ref:
    zip_ref.extractall(filename)


for dataset in kaggle_datasets:
  unzip(dataset)

# Time to make a dataset directory and move all the other contents into it.
import shutil
import os

TARGET = "./dataset"

if not os.path.exists(TARGET):
  os.makedirs(TARGET)

 
def move(source_dir, target_dir):
  file_names = os.listdir(source_dir)
    
  for file_name in file_names:
    shutil.move(os.path.join(source_dir, file_name), target_dir)


paths = [
  "./anime-faces/data",
  "./animefacedataset/images",
  "./another-anime-face-dataset/animefaces256cleaner"
]

for path in paths:
  move(path, TARGET)

# Get the count of file
print(len(os.listdir(TARGET)))

