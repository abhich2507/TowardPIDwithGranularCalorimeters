


import os
import shutil

# Define the list of granularities
granularities = [
    "100_100_10", "100_100_25", "10_10_10", "10_10_25",
    "25_25_10", "25_25_25", "50_50_10", "50_50_25",
    "100_100_100", "100_100_50", "10_10_100", "10_10_50",
    "25_25_100", "25_25_50", "50_50_100", "50_50_50"
]

# Directory where your files are currently located
source_directory = "/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/"

# Directory where you want to store the new granularity folders
target_base_directory = "/home/alma1/GNN/Deepset/28oct_DNN/results/"

# Iterate through granularities
for granularity in granularities:
    # Create a directory for each granularity in the new target base directory
    target_directory = os.path.join(target_base_directory, granularity)
    os.makedirs(target_directory, exist_ok=True)
    
    # Move files that match the granularity pattern to the corresponding directory
    for file_name in os.listdir(source_directory):
        if granularity in file_name:  # Check if granularity is in the file name
            source_file = os.path.join(source_directory, file_name)
            target_file = os.path.join(target_directory, file_name)
            shutil.move(source_file, target_file)

print("Files have been organized into their corresponding directories in the target base directory.")


