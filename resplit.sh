#!/bin/bash

# Define directories
original_test_dir="/home/michael/Desktop/ifile/serverstuff/hsi/test"             # Replace with the path to your original test directory
original_validation_dir="/home/michael/Desktop/ifile/serverstuff/hsi/validation" # Replace with the path to your original validation directory
processed_dir="/home/michael/Desktop/ifile/nn/resizedstuff/mst_resized" # Replace with the path to your processed directory
new_test_dir="$processed_dir/mst_resized_test"     # New directory for test split
new_validation_dir="$processed_dir/mst_resized_validation" # New directory for validation split

# Create new test and validation split directories if they don't exist
mkdir -p "$new_test_dir"
mkdir -p "$new_validation_dir"

# Function to move files based on the original split
move_files() {
  local original_dir="$1"
  local new_dir="$2"
  
  # Loop through each species directory in the original split
  for species in "$original_dir"/*; do
    species_name=$(basename "$species")
    mkdir -p "$new_dir/$species_name"
    
    # Loop through each file in the species directory
    for file in "$species"/*; do
      filename=$(basename "$file")
      processed_file="$processed_dir/$species_name/${filename%.png}_resized.mat"
      
      # Check if the processed file exists and move it
      if [ -f "$processed_file" ]; then
        mv "$processed_file" "$new_dir/$species_name/"
      fi
    done
  done
}

# Move test files
move_files "$original_test_dir" "$new_test_dir"

# Move validation files
move_files "$original_validation_dir" "$new_validation_dir"

echo "Test and validation files moved to new split directories."

