import numpy as np
import pandas as pd
import os, random, cv2
from mxnet.gluon.data.vision import transforms
# from mxnet.gluon.data import RecordFileDataset
from mxnet import gluon, image
from mxnet.io import ImageRecordIter
import subprocess
from io import StringIO
from Im2Rec import Im2Rec

class DataRecHandler:
    def __init__(self, root_path, rank, batch_size, num_workers, k, create_recs=False, oversample_technique='transformed'):
        # Parameters
        # path = root path
        # rank is used as /path/rank
        # for k-fold cross validation
        self.chunks = []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k
        self.rank = rank

        self.val_id =0
        self.test_id = 1
        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.root_path = root_path
        self.rank_path = os.path.join(root_path, rank)

        # class count -> crashes if there are files in the rank_path and not only folders
        self.classes = len([x[0] for x in os.walk(self.rank_path)])-1
        subdirs = os.listdir(self.rank_path)
        # subdirs = [x[0] for x in os.walk(self.rank_path)]
        self.samples_per_class = {}
        for subdir in subdirs:
            if os.path.isdir(os.path.join(self.rank_path,subdir)):
                self.samples_per_class[subdir] = len(os.listdir(os.path.join(self.rank_path, subdir)))

        # oversample and create rec lists
        self._oversample(technique=oversample_technique)
        self._create_recordIO_lists()


        # self._load_ids(root_path, batch_size, num_workers, augment)
        # self._set_data_sets(batch_size=batch_size,fold=fold,num_workers=num_workers)

    def _oversample(self, technique = 'transformed'):
        # technique = 'naive'
        # technique = 'transformed'
        # technique = 'smote' not implemented yet
        if technique == 'naive':
            max_class_count = self.samples_per_class[max(self.samples_per_class, key=self.samples_per_class.get)]
            for subdir in self.samples_per_class:
                subdir_path = os.path.join(self.rank_path,subdir)
                files = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path)]
                class_count = self.samples_per_class[subdir]
                # generate new transformed images of a randomly drawn image until the amount matches that of the max class
                for i in range(max_class_count - class_count):
                    if len(files)>0:
                        rdn_image_file = random.sample(files, 1)[0]
                        rdn_image = image.imread(rdn_image_file)
                        file_name = rdn_image_file[:-4] + '__oversampled_naive__' + str(i) + '.png'
                        cv2.imwrite(file_name, rdn_image.asnumpy())
        if technique == 'transformed':
            max_class_count = self.samples_per_class[max(self.samples_per_class, key=self.samples_per_class.get)]
            for subdir in self.samples_per_class:
                subdir_path = os.path.join(self.rank_path,subdir)
                files = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path)]
                class_count = self.samples_per_class[subdir]
                # generate new transformed images of a randomly drawn image until the amount matches that of the max class
                print(subdir_path)
                print('class count '+str(class_count))
                print('max class count ' + str(max_class_count))
                for i in range(max_class_count - class_count):
                    if len(files)>0:
                        rdn_image_file = random.sample(files, 1)[0]
                        rdn_image = image.imread(rdn_image_file)
                        transformer = self.transform(mode='resample')
                        transformed_image = transformer(rdn_image)
                        file_name = rdn_image_file[:-4] + '__oversampled_transformed__' + str(i) + '.png'
                        cv2.imwrite(file_name, transformed_image.asnumpy())
        if technique == 'smote':
            pass


    def _create_recordIO_lists(self):
        # creates lists for and RecordIO files in chunks of k
        # and loads the chunks to self.chunks
        # arguments for im2rec.py
        im2rec = os.path.join(self.root_path, 'im2rec.py')
        file_prefix = self.rank_path + '/' + self.rank
        # invoke im2rec to create lists for RecordIO creation
        # capture output as mapping from id to taxon
        # tha mapping is the same across chunks
        # mapping = subprocess.check_output([im2rec, file_prefix, self.rank_path, '--recursive', '--list', '--pack-label', '--chunks', str(self.k),'--num-thread', str(self.num_workers)])
        i2r = Im2Rec([file_prefix, self.rank_path, '--recursive', '--list', '--pack-label', '--chunks', str(self.k),'--num-thread', str(self.num_workers)])
        raw_data = StringIO(i2r.str_mapping)
        mapping_df = pd.read_csv(raw_data, sep=' ', names=['taxon', 'id'], header=None)
        if self.rank == 'all-in-one': # add multi-labels
            # create mapping dicts
            id2taxon = mapping_df.to_dict()['taxon']
            taxon2id = {v: k for k, v in id2taxon.items()}

            missing_values = ['', 'unknown', 'unclassified', 'unidentified']
            taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
            csv_path = os.path.join(self.root_path, 'im.merged.v10032020_unique_id_set.csv')
            df = pd.read_csv(csv_path, na_values=missing_values)[taxonomic_groups]
            for fold in range(self.k):
                # load list as df
                list_name = self.rank+'_'+str(fold)+'.lst'
                list_df = pd.read_csv(os.path.join(self.rank_path,list_name), sep='\t', names=['id', 'label', 'file'], header=None)
                new_list = ""
                for index, row in list_df.iterrows():
                    new_list = new_list+str(row['id'])
                    taxon = id2taxon[row['label']]
                    higher_taxons = df.loc[df['species']==taxon].iloc[0,:]
                    for item in higher_taxons.to_list():#add additional labels
                        if item is np.nan:
                            item = 'nan'
                            # print('adding '+str(item))
                        if item not in taxon2id:
                            taxon2id[item] = len(taxon2id)+1
                        # print('%s with id %i '%(item,taxon2id[item]))
                        new_list = new_list + '\t'+str(taxon2id[item])
                    new_list = new_list+ '\t'+ row['file']+'\n'
                    fn = self.rank+'_'+str(fold)+'.lst'
                    # print('##########')
                with open(os.path.join(self.rank_path, fn), 'wt') as out_file:
                    out_file.write(new_list)
            mapping_df = pd.DataFrame(taxon2id.items())
        #######

        mapping_df.to_csv(os.path.join(self.rank_path,'mapping.csv'))









    def load_rec(self, fold):
        # invoke im2rec to create RecordIO according to self.k
        #
        # load chunk recs into dataloaders for train, test, val
        im2rec = os.path.join(self.root_path, 'im2rec.py')
        file_prefix = self.rank_path + '/' + self.rank
        ###########################################

        fout = file_prefix+'_train_'+str(fold)+'.lst'
        with open(fout, 'w') as outfile:
            for i in range(self.k):
                if i != fold: # leave validation list out
                    fin = file_prefix+'_'+str(i)+'.lst'
                    with open(fin) as infile:
                        outfile.write(infile.read())

        # create record for train and test
        fout = file_prefix+'_train_'+str(fold)+'.lst'
        print('creating '+fout)
        # subprocess.run([im2rec, fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
        i2r = Im2Rec([fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
        list_arg = file_prefix+'_'+str(fold)+'.lst'
        print('creating '+list_arg)
        # subprocess.run([im2rec, list_arg, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
        i2r = Im2Rec([list_arg, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
        label_width = 1
        if self.rank == 'all-in-one':
            label_width = 6

        rec_path = file_prefix+'_train_'+str(fold)+'.rec'
        idx_path = file_prefix+'_train_'+str(fold)+'.idx'
        lst_path = file_prefix+'_train_'+str(fold)+'.lst'
        self.train_data = ImageRecordIter(
            path_imgrec=rec_path,
            path_imgidx=idx_path,
            path_imglist=lst_path,
            data_shape=(3, 224, 224),
            batch_size=self.batch_size,
            # batch_size=batch_size,
            label_width=label_width,
            shuffle=True #train true, test no
        )
        # self.train_data = gluon.data.DataLoader(RecordFileDataset(rec_path),batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        rec_path = file_prefix+'_'+str(fold)+'.rec'
        idx_path = file_prefix+'_'+str(fold)+'.idx'
        lst_path = file_prefix+'_'+str(fold)+'.lst'
        # self.test_data = gluon.data.DataLoader(RecordFileDataset(rec_path),batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_data = ImageRecordIter(
            path_imgrec=rec_path,
            path_imgidx=idx_path,
            path_imglist=lst_path,
            data_shape=(3, 224, 224),
            batch_size=self.batch_size,
            # batch_size=batch_size,
            label_width=label_width,
            shuffle=True #train true, test no
        )
        pass


    # def _load_ids(self, path, batch_size , num_workers, augment):
    #
    #     ### load imagedataset (ids)
    #     train_path = os.path.join(path, 'train')
    #     val_path = os.path.join(path, 'val')
    #     test_path = os.path.join(path, 'test')
    #
    #     subdirs = os.listdir(train_path)
    #     self.classes = len([subdir for subdir in subdirs if os.path.isdir(os.path.join(train_path, subdir))])
    #
    #     self.samples_per_class = {}
    #     for subdir in subdirs:
    #         self.samples_per_class[subdir] = len(os.listdir(os.path.join(train_path, subdir)))
    #
    #     # resample class distribution in directory
    #     print("\t\t\tattempting to resample distribution with pseudo bootstrapping")
    #     self.resample_class_distribution_in_directory(train_path)
    #     self.resample_class_distribution_in_directory(test_path)
    #     self.resample_class_distribution_in_directory(val_path)
    #
    #     self.samples_per_class_normalized = {}
    #     for subdir in subdirs:
    #         self.samples_per_class_normalized[subdir] = len(os.listdir(os.path.join(train_path, subdir)))
    #
    #     print("\t\t\t\tloading train as IFDS")
    #     train_ids = gluon.data.vision.ImageFolderDataset(train_path)
    #     print("\t\t\t\tloading test as IFDS")
    #     test_ids = gluon.data.vision.ImageFolderDataset(test_path)
    #     print("\t\t\t\tloading val as IFDS")
    #     val_ids = gluon.data.vision.ImageFolderDataset(val_path)
    #
    #     print('\t\t\t\tIFDS -> DataLoader, augmentation: %s' %augment)
    #     # split into train, vla, test
    #     if augment == 'transform':
    #         # transform_first transforms only the data[0] but not the label[1]
    #         self.train_data = gluon.data.DataLoader(
    #             train_ids.transform_first(self.transform()),
    #             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #         self.val_data = gluon.data.DataLoader(
    #             val_ids.transform_first(self.transform(mode='test')),
    #             batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #         self.test_data = gluon.data.DataLoader(
    #             test_ids.transform_first(self.transform(mode='test')),
    #             batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #     elif augment == 'resize':
    #         self.train_data = gluon.data.DataLoader(
    #             train_ids.transform_first(self.resize()),
    #             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #         self.val_data = gluon.data.DataLoader(
    #             val_ids.transform_first(self.resize()),
    #             batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #         self.test_data = gluon.data.DataLoader(
    #             test_ids.transform_first(self.resize()),
    #             batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #     else:
    #         self.train_data = gluon.data.DataLoader(
    #             train_ids,
    #             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #         self.val_data = gluon.data.DataLoader(
    #             val_ids,
    #             batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #         self.test_data = gluon.data.DataLoader(
    #             test_ids,
    #             batch_size=batch_size, shuffle=False, num_workers=num_workers)


    def transform(self, jitter_param = 0.4, lighting_param = 0.1, mode = 'train'):
        if mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomFlipLeftRight(),
                # transforms.RandomFlipTopBottom(),
                # transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                #                              saturation=jitter_param),
                # transforms.RandomLighting(lighting_param),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            transform = transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif mode == 'resample':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomFlipLeftRight(),
                transforms.RandomFlipTopBottom(),
                transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                             saturation=jitter_param),
                transforms.RandomLighting(lighting_param)
            ])
        return transform

    # def resize(self):
    #     transform = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         # transforms.Resize(256),
    #         # transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    #     return transform

    # def augment_ids(self):
    #     print("")
