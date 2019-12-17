import numpy as np
import pandas as pd
import os, random, cv2
from mxnet.gluon.data.vision import transforms
from mxnet import gluon, image

class DataHandler:
    def __init__(self, path, batch_size , num_workers, transform):
        self.train_data = []
        self.test_data = []
        self.validate_data = []

        # if set == "caltech256":
        #     self._load_caltech256_ids(path, batch_size , num_workers, transform)
        # elif set == "minc2500":
        #     self._load_minc2500_ids(path, batch_size , num_workers, transform)
        # elif set == "fungi":
        self._load_ids(path, batch_size , num_workers, transform)

    def _load_ids(self, path, batch_size , num_workers, transform):

        ### load imagedataset (ids)
        train_path = os.path.join(path, 'train')
        val_path = os.path.join(path, 'val')
        test_path = os.path.join(path, 'test')
        train_ids = gluon.data.vision.ImageFolderDataset(train_path)
        test_ids = gluon.data.vision.ImageFolderDataset(test_path)
        val_ids = gluon.data.vision.ImageFolderDataset(val_path)

        subdirs = os.listdir(train_path)
        self.classes = len([subdir for subdir in subdirs if os.path.isdir(os.path.join(train_path, subdir))])

        self.samples_per_class = {}
        for subdir in subdirs:
            self.samples_per_class[subdir] = len(os.listdir(os.path.join(train_path, subdir)))

        # resample class distribution in directory
        self.resample_class_distribution_in_directory(train_path)
        self.resample_class_distribution_in_directory(test_path)
        self.resample_class_distribution_in_directory(val_path)

        self.samples_per_class_normalized = {}
        for subdir in subdirs:
            self.samples_per_class_normalized[subdir] = len(os.listdir(os.path.join(train_path, subdir)))

        # split into train, vla, test
        if transform:
            self.train_data = gluon.data.DataLoader(
                train_ids.transform_first(self.transform()),
                batch_size=batch_size, shuffle=True, num_workers=num_workers)

            self.val_data = gluon.data.DataLoader(
                val_ids.transform_first(self.transform(mode='test')),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)

            self.test_data = gluon.data.DataLoader(
                test_ids.transform_first(self.transform(mode='test')),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            self.train_data = gluon.data.DataLoader(
                train_ids,
                batch_size=batch_size, shuffle=True, num_workers=num_workers)

            self.val_data = gluon.data.DataLoader(
                val_ids,
                batch_size=batch_size, shuffle=False, num_workers=num_workers)

            self.test_data = gluon.data.DataLoader(
                test_ids,
                batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def resample_class_distribution_in_directory(self, path):
        max_class_count = self.samples_per_class[max(self.samples_per_class, key=self.samples_per_class.get)]
        for subdir in self.samples_per_class:
            subdir_path = os.path.join(path,subdir)
            files = [os.path.join(subdir_path, file) for file in os.listdir(os.path.join(path, subdir))]
            class_count = self.samples_per_class[subdir]
            # generate new transformed images of a randomly drawn image until the amount matches that of the max class
            for i in range(max_class_count - class_count):
                if len(files)>0:
                    rdn_image_file = random.sample(files, 1)[0]
                    rdn_image = image.imread(rdn_image_file)
                    transformer = self.transform(mode='resample')
                    transformed_image = transformer(rdn_image)
                    file_name = rdn_image_file[:-4] + '_transformed_' + str(i) + '.png'
                    cv2.imwrite(file_name, transformed_image.asnumpy())

    def transform(self, jitter_param = 0.4, lighting_param = 0.1, mode = 'train'):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomFlipLeftRight(),
                transforms.RandomFlipTopBottom(),
                transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                             saturation=jitter_param),
                transforms.RandomLighting(lighting_param),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif mode == 'resample':
            transform = transforms.Compose([
                transforms.RandomFlipLeftRight(),
                transforms.RandomFlipTopBottom(),
                transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                             saturation=jitter_param),
                transforms.RandomLighting(lighting_param)
            ])
        return transform

    def augment_ids(self):
        print("")
