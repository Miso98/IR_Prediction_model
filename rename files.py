import os

def rename_images(directory):
    files = os.listdir(directory)
    for i, file in enumerate(files):
        if file.endswith('.png'):  # Ensures only PNG files are renamed
            new_name = f"image{i+1}.png"  # Renames to image1.png, image2.png, etc.
            os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
            print(f"Renamed {file} to {new_name}")

# Rename files in the 'regular' and 'irregular' folders
rename_images('/home/mitchell/Documents/repos/IR_project/dataset/train/regular')
rename_images('/home/mitchell/Documents/repos/IR_project/dataset/train/irregular')
