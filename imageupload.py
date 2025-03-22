import os
import random
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Disable the root window that Tkinter opens
Tk().withdraw()

# Ask user to select an image file
image_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])

if image_path:
    # Create a folder with a random number name
    random_folder_name = str(random.randint(1000, 9999))
    folder_path = os.path.join('imagefolder', random_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(random_folder_name)
    # Generate a random name for the image
    random_image_name = str(random_folder_name) + os.path.splitext(image_path)[1]
    print(random_image_name)
    # Copy the image to the newly created folder with the random name
    save_path = os.path.join(folder_path, random_image_name)
    shutil.copy(image_path, save_path)
    
    print(f"Image saved successfully at: {save_path}")
else:
    print("No image selected.")