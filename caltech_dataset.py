from torchvision.datasets import VisionDataset

from PIL import Image

from sklearn.model_selection import train_test_split

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    classes = {} # it contains the mappings class_name:numeric_label
    excluded = ['BACKGROUND_Google']
    class_num = 0 # counts how many classes we have

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        self.images = []
        self.labels = []

        with open(f"./Caltech101/{split}.txt", 'r') as paths:
            for line in paths.readlines():
                class_name = line[:line.find('/')]
                if (class_name not in Caltech.excluded) and (class_name not in Caltech.classes):
                    Caltech.classes[class_name] = Caltech.class_num
                    Caltech.class_num += 1
                numeric_label = Caltech.classes[class_name]
                self.labels.append(numeric_label)

                img = pil_loader(f"{root}/{line[:-1]}")
                self.images.append(img)

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class)
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.images[index], self.labels[index] # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.labels) # Provide a way to get the length (number of elements) of the dataset
        return length


    def stratified_split_indexes(train_size=0.5):
        indexes = range(len(labels))
        train_i, val_i = train_test_split(train_size=train_size, random_state=42, stratify=labels)
        return train_i, val_i
