import pandas as pd
from pathlib import Path
from os.path import join
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
import cv2
import numpy as np

class PandasDataset(Dataset):

    def __init__(self, dataframe, img_column='img', label_column='label', transform=None, loader=default_loader):
        """
        :param dataframe: Pandas DataFrame.
        :param img_column: Pandas column name for image path.
        :param label_column: Pandas column name for label.
        :param transform: PyTorch transformations to apply to each image.
        """

        # Image loader
        self.loader = loader

        self.data = dataframe
        assert isinstance(self.data, pd.DataFrame), 'Argument `dataframe` should be an instance of Pandas DataFrame.'

        # Column names
        self.img_column = img_column
        self.label_column = label_column

        # PyTorch transformations
        self.transform = transform

        # Get a list of images
        self.imgs = self.data[[self.img_column, self.label_column]].to_dict(orient='index').items()

        self.imgs = [tuple(v.values()) for k, v in self.imgs]

        # List of unique classes in our dataset
        self.classes = list(sorted(self.data[self.label_column].unique().tolist()))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

    def __getitem__(self, index):

        # Get an image path and label based on index
        path, target = self.imgs[index]

        # Convert class name to its index
        target = self.class_to_idx[target]

        # Read an image
        img = cv2.imread(join('/opt', path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32'))

        if self.transform is not None:
            # Apply transformations
            img = self.transform(img)

        return path, img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        output = 'Dataset ' + self.__class__.__name__ + '\n'
        output += '    Number of data points: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        output += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return output
