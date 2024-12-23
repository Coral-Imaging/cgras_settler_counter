import os
import shutil

# Define paths
labels_folder = "/media/wardlewo/cslics_ssd/SCU_Pdae_Data/split and tilling/train/labels"
images_folder = "/media/wardlewo/cslics_ssd/SCU_Pdae_Data/split and tilling/train/images"
destination_folder = "/media/wardlewo/cslics_ssd/SCU_Pdae_Data/reduced_negs_dataset/train"

# Ensure the destination folders exist
os.makedirs(destination_folder, exist_ok=True)

# Get lists of label files
label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

# Separate non-empty and empty label files
non_empty_labels = [f for f in label_files if os.path.getsize(os.path.join(labels_folder, f)) > 0]
empty_labels = [f for f in label_files if os.path.getsize(os.path.join(labels_folder, f)) == 0]
print("Number of non-empty labels found: ",len(non_empty_labels))
print("Number of empty labels found: ",len(empty_labels))
# Ensure equal counts of non-empty and empty labels
selected_empty_labels = empty_labels[:len(non_empty_labels)]

# Combine the selected label files
selected_labels = non_empty_labels + selected_empty_labels
print("Total number of labels in new dataset: ",len(selected_labels))
# Process files
for label_file in selected_labels:
    label_path = os.path.join(labels_folder, label_file)
    
    # Find corresponding image file
    image_file = label_file.replace('.txt', '.jpg')
    image_path = os.path.join(images_folder, image_file)
    
    if os.path.exists(image_path):
        # Define destination paths
        label_dest = os.path.join(destination_folder, "labels", label_file)
        image_dest = os.path.join(destination_folder, "images", image_file)
        
        # Create subdirectories if not exist
        os.makedirs(os.path.dirname(label_dest), exist_ok=True)
        os.makedirs(os.path.dirname(image_dest), exist_ok=True)
        
        # Copy files to destination
        shutil.copy(label_path, label_dest)
        shutil.copy(image_path, image_dest)
    else:
        print(f"Image not found for label: {label_file}")

print("File transfer complete.")
