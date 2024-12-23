import os
from pathlib import Path
from tqdm import tqdm

def change_class_to_zero(label_dir, output_dir):
    """
    Modifies all label files to set the class ID to 0.

    Args:
        label_dir (str): Path to the folder containing the label files.
        output_dir (str): Directory to save modified label files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through label files
    for label_file in tqdm(sorted(Path(label_dir).glob("*.txt")), desc="Processing labels"):
        # Read the label file
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Modify class ID in each line
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure it has at least a class ID and bounding box/polygon data
                parts[0] = "0"  # Change the class ID to 0
                updated_lines.append(" ".join(parts))
            else:
                print(f"Skipping malformed line in {label_file}: {line}")

        # Write the updated lines to the output directory
        output_path = Path(output_dir) / label_file.name
        with open(output_path, "w") as f:
            f.write("\n".join(updated_lines) + "\n")

    print(f"Processing complete. Modified label files saved in: {output_dir}")


# Example usage
if __name__ == "__main__":
    label_dir = "/media/wardlewo/cslics_ssd/cgras_datasets/Seg+ClassTester/Seg_test/test/labels"  # Path to original labels
    change_class_to_zero(label_dir, label_dir)
