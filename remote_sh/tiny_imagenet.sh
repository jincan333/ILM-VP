#!/bin/bash

# Replace DOWNLOAD_LINK with the link to the dataset
DOWNLOAD_LINK="https://image-net.org/data/tiny-imagenet-200.zip"

# The directory where the script will store the dataset
STORAGE_DIR="dataset/tiny_imagenet"

# Check if the directory exists, otherwise create it
mkdir -p ${STORAGE_DIR}

cd ${STORAGE_DIR}

# Download and unzip in one line
wget -O tiny-imagenet-200.zip ${DOWNLOAD_LINK} && unzip tiny-imagenet-200.zip