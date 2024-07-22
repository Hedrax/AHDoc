import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to the dataset and labels
BASE_DIR = './dataset/'
LABELS_FILE = os.path.join(BASE_DIR, 'labels.txt')

# Paths for split datasets
TRAIN_DIR = os.path.join(BASE_DIR, 'train/')
VAL_DIR = os.path.join(BASE_DIR, 'val/')
TEST_DIR = os.path.join(BASE_DIR, 'test/')

# Ensure directories exist
for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)  # Clear it if it exists
        os.makedirs(directory)

# Read labels file
with open(LABELS_FILE, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Use only the first 1 million labels
max_samples = 1000000
lines = lines[:max_samples]

# Calculate total samples
total_samples = len(lines)

# Function to print the total samples
def print_total_samples(total_samples):
    print(f"Total samples: {total_samples}")

# Split dataset
train_val_lines, test_lines = train_test_split(lines, test_size=0.1, random_state=42)
train_lines, val_lines = train_test_split(train_val_lines, test_size=0.1, random_state=42)

# Helper function to copy files and write labels
def copy_files_and_write_labels(lines, target_dir):
    labels_path = os.path.join(target_dir, 'labels.txt')
    with open(labels_path, 'w', encoding='utf-8') as labels_file:
        for idx, line in enumerate(lines):
            img_filename = line.split('_')[0] + '.png'
            img_src_path = os.path.join(BASE_DIR, img_filename)
            img_dst_path = os.path.join(target_dir, img_filename)
            if os.path.exists(img_src_path):
                shutil.copy(img_src_path, img_dst_path)
                labels_file.write(line)
            else:
                print(f"Warning: {img_src_path} does not exist and will not be copied.")
            
            # Print debug message every 1000 lines
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} files for {target_dir}")

# Copy files and write labels for each set
print("Copying training data...")
copy_files_and_write_labels(train_lines, TRAIN_DIR)
print("Copying validation data...")
copy_files_and_write_labels(val_lines, VAL_DIR)
print("Copying test data...")
copy_files_and_write_labels(test_lines, TEST_DIR)

print("Dataset split complete!")

# Print total samples
print_total_samples(total_samples)

# Update config.py
config_path = 'config.py'
with open(config_path, 'r', encoding='utf-8') as file:
    config_lines = file.readlines()

with open(config_path, 'w', encoding='utf-8') as file:
    for line in config_lines:
        if line.strip().startswith('TOTAL_SAMPLES'):
            file.write(f'TOTAL_SAMPLES = {total_samples}\n')
        else:
            file.write(line)
