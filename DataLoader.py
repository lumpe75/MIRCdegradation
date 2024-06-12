import zipfile
from torch.utils.data import Dataset
from PIL import Image
import os

def extract_zip_files(zip_dir, extract_dir):
    """
    Extracts all zip files in a directory to a specified directory.

    :param zip_dir: Directory containing zip files
    :param extract_dir: Directory where the files will be extracted
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    for zip_filename in os.listdir(zip_dir):
        if zip_filename.endswith(".zip"):
            zip_path = os.path.join(zip_dir, zip_filename)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subdirectories by label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}

        # Walk through the root directory and list images
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                self.label_map[label] = len(self.label_map)  # Create a mapping from label to index
                for image_name in os.listdir(label_path):
                    if image_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(label_path, image_name))
                        self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label





if __name__ == '__main__':
    zip_dir = 'C:\\Users\\Lumpe\\Pictures\\Imagenet'
    extract_dir = 'C:\\Users\\Lumpe\\Pictures\\Imagenet'
    extract_zip_files(zip_dir, extract_dir)