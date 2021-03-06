import numpy as np
import pandas as pd
import os, random, cv2, shutil
from mxnet.gluon.data.vision import transforms
# from mxnet.gluon.data import RecordFileDataset
from mxnet import gluon, image
from mxnet.io import ImageRecordIter
import mxnet as mx
from io import StringIO
from Im2Rec import Im2Rec

class DataRecHandler:
    def __init__(self, root_path, rank_name, rank_idx, batch_size, num_workers, k, create_recs=False, oversample_technique='transformOversampled'):
        # Parameters
        # path = root path
        # rank is used as /path/rank
        # for k-fold cross validation
        self.chunks = []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k
        self.rank = rank_name
        self.rank_idx = rank_idx

        self.val_id =0
        self.test_id = 1
        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.test_ratio = 0.3
        self.train_ratio = 0.7

        # self.mode ='_orig_tt-split_SL'
        self.mode ='_orig_tt-split_ML'
        # self.mode ='_oversampled_tt-split_SL'
        # self.mode ='_oversampled_tt-split_ML'
        # self.mode ='_orig_xval_SL'
        # self.mode ='_orig_xval_ML'
        # self.mode ='_oversampled_xval_SL'
        # self.mode ='_oversampled_xval_ML'

        self.root_path = root_path

        csv_path = os.path.join(self.root_path, 'im.merged.v10032020_unique_id_set.csv')
        # 1) set up one global unique mapping df, containing one col with all taxons and one with all ids -> self._get_global_mapping
        self._build_global_mapping(csv_path)

        if self.mode.split('_')[-2]=='tt-split' and self.mode.split('_')[-3]=='oversampled':
            # creating train separately
            self.rank_path = os.path.join(root_path, rank_name, 'train')
            self.file_prefix = self.rank_path + '/' + self.rank + '_train'  + self.mode
            if create_recs: # create train and test folders
                self.sample()
            self._get_samples_p_class()
            if create_recs:
                self._oversample(technique=oversample_technique)#works with samples_per_class and rank_path
                self._create_recordIO()

            # creating test separately
            self.rank_path = os.path.join(root_path, rank_name, 'test')
            self.file_prefix = self.rank_path + '/' + self.rank + '_test' + self.mode
            self._get_samples_p_class()
            if create_recs:
                # self._oversample(technique=oversample_technique) !!! VERY IMPORTANT TO NOT OVERSAMPLE TEST, AS IT BIASES PCC
                self._create_recordIO()
            self.rank_path = os.path.join(root_path, rank_name)
            self.file_prefix = self.rank_path + '/' + self.rank + self.mode
            self._get_samples_p_class()
        elif self.mode.split('_')[-2] == 'xval' and self.mode.split('_')[-3] == 'oversampled': #!!! VERY IMPORTANT TO NOT OVERSAMPLE TEST, AS IT BIASES THE PCC
            max_class_count = 0
            for i in range(k):
                # creating X0 separately
                self.rank_path = os.path.join(root_path, rank_name, 'x%s_train'%i)
                if create_recs and i == 0:  # create all X folders
                    self.sampleX() # create chunk folders for train and test, so that train can be oversampled while test will not
                self._get_samples_p_class()
                max_class_count = max(max_class_count, self.samples_per_class[max(self.samples_per_class, key=self.samples_per_class.get)])
            if create_recs:
                for i in range(k):
                    self.rank_path = os.path.join(root_path, rank_name, 'x%s_train' % i)
                    self._get_samples_p_class()
                    ######### commment out for HC-orig
                    self._oversample(technique=oversample_technique,max_class_count=max_class_count)
                    ######### commment out for HC-orig
            self.rank_path = os.path.join(root_path, rank_name)
            self._get_samples_p_class()
            self.file_prefix = self.rank_path + '/' + self.rank + self.mode
            self._create_recordIO()
        else:
            self.rank_path = os.path.join(root_path, rank_name)
            self.file_prefix = self.rank_path + '/' + self.rank + self.mode
            self._get_samples_p_class()
            # oversample and create rec lists
            if create_recs:
                if self.mode.split('_')[-3] != 'orig':
                    self._oversample(technique=oversample_technique)
                self._create_recordIO()


    def _get_samples_p_class(self):
        subdirs = os.listdir(self.rank_path)
        # subdirs = [x[0] for x in os.walk(self.rank_path)]
        self.samples_per_class = {}
        for subdir in subdirs:
            if os.path.isdir(os.path.join(self.rank_path, subdir)):
                self.samples_per_class[subdir] = len(os.listdir(os.path.join(self.rank_path, subdir)))

    def _oversample(self, technique='transformOversampled', max_class_count=None):
        # technique = 'naiveOversampled'
        # technique = 'transformed'
        # technique = 'smote' not implemented yet
        if max_class_count == None:
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
                    transformer = self.transform(mode=technique)
                    transformed_image = transformer(rdn_image)
                    file_name = rdn_image_file[:-4] + '__'+ technique +'__' + str(i) + '.png'
                    cv2.imwrite(file_name, transformed_image.asnumpy())
        if technique == 'smote':
            pass

    def sampleX(self):
        # Create directories:
        # already exists: path/
        # path/cuts/X0...
        # path/cuts/test/
        # path/cuts/val/
        # sample images in
        path = os.path.join(self.root_path, self.rank)
        x_images = dict()

        # all subdirs in cut_path without files
        subdirs = [os.path.join(path, x) for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))]
        subdirs_names = [x for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))]
        # all files in subdirs
        filenames = []
        for subdir in subdirs:
            files_subdir = os.listdir(subdir)
            full_file_names = [os.path.join(subdir, f) for f in files_subdir]
            filenames.append(full_file_names)
        filenames = [item for sublist in filenames for item in sublist]
        for i in range(self.k-1):
            x_images[i] = random.sample(filenames, round(len(filenames) * 1 / self.k))
            filenames = [elem for elem in filenames if elem not in x_images[i]]
        x_images[self.k-1] = random.sample(filenames, round(len(filenames)))
        # x_images[0] = random.sample(filenames, round(len(filenames) * 1 / self.k))
        # filenames = [elem for elem in filenames if elem not in x_images[0]]
        # x_images[1] = random.sample(filenames, round(len(filenames) * 1 / self.k))
        # filenames = [elem for elem in filenames if elem not in x_images[1]]
        # x_images[2] = random.sample(filenames, round(len(filenames) * 1 / self.k))
        # filenames = [elem for elem in filenames if elem not in x_images[2]]
        # x_images[3] = random.sample(filenames, round(len(filenames) * 1 / self.k))
        # filenames = [elem for elem in filenames if elem not in x_images[3]]
        # x_images[4] = random.sample(filenames, round(len(filenames)))
        for tt in ['_train','_test']:
            for i in range(self.k):
                x_path = os.path.join(self.root_path, self.rank, 'x%s%s' %(i,tt))
                if not os.path.exists(x_path):
                    print('\tsampling into directories: %s'%x_path)
                    # if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
                    # creating x set
                    os.makedirs(x_path)
                    for l in subdirs_names:
                        os.makedirs(os.path.join(x_path, l))
                    # Copy files to corresponding directory
                    for file in x_images[i]:
                        split_list = file.split('/')
                        label = split_list[-2]
                        to_path = os.path.join(x_path, label)
                        print('copying %s to %s' %(file, to_path))
                        shutil.copy(file, to_path)

    def sample(self):
        # Create directories:
        # already exists: path/cuts/
        # path/cuts/train/
        # path/cuts/test/
        # path/cuts/val/
        # sample images in
        path = os.path.join(self.root_path, self.rank)

        print('\tsampling into directories: train, test, val')
        train_path = os.path.join(path, 'train')
        # val_path = os.path.join(path, 'val')
        test_path = os.path.join(path, 'test')

        # if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            # creating train, validation and test set
            # all subdirs in cut_path without files
            subdirs = [os.path.join(path, x) for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))]
            subdirs_names = [x for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))]
            # all files in subdirs
            filenames = []
            for subdir in subdirs:
                files_subdir = os.listdir(subdir)
                full_file_names = [os.path.join(subdir, f) for f in files_subdir]
                filenames.append(full_file_names)


            # # list all files in subdirs
            # filenames = [os.path.join(x, os.listdir(x)) for x in subdirs]
            # # make it full path
            filenames = [item for sublist in filenames for item in sublist]
            train_images = random.sample(filenames, round(len(filenames) * self.train_ratio))
            filenames = [elem for elem in filenames if elem not in train_images]
            test_images = random.sample(filenames, round(len(filenames) ))
            # test_images = random.sample(filenames, round(len(filenames) * self.test_ratio))
            # filenames = [elem for elem in filenames if elem not in val_images]
            # test_images = filenames

            os.makedirs(train_path)
            # os.makedirs(val_path)
            os.makedirs(test_path)

            for l in subdirs_names:
                os.makedirs(os.path.join(train_path, l))
                # makedirs(os.path.join(val_path, l))
                os.makedirs(os.path.join(test_path, l))

            # Copy files to corresponding directory
            for file in train_images:
                split_list = file.split('/')
                label = split_list[-2]
                to_path = os.path.join(train_path, label)
                shutil.copy(file, to_path)

            # for file in val_images:
            #     split_list = file.split('/')
            #     label = split_list[-2]
            #     to_path = os.path.join(val_path, label)
            #     shutil.copy(file, to_path)

            for file in test_images:
                split_list = file.split('/')
                label = split_list[-2]
                to_path = os.path.join(test_path, label)
                shutil.copy(file, to_path)

    def __deprecated__create_ml_list(self, mapping_df):
        # create mapping dicts
        id2taxon = mapping_df.to_dict()['taxon']
        # taxon2id = {v: k for k, v in id2taxon.items()}
        taxon2id = dict()
        missing_values = ['', 'unknown', 'unclassified', 'unidentified']
        taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        csv_path = os.path.join(self.root_path, 'im.merged.v10032020_unique_id_set.csv')
        df = pd.read_csv(csv_path, na_values=missing_values)[taxonomic_groups]
        self.classes = dict()

        if self.file_prefix.split('_')[-2] == 'xval' and self.file_prefix.split('_')[-3] == 'oversampled':
            mapping_part = dict()
            for tt in ['_train','_test']:
                for fold in range(self.k):
                    file_prefix = self.file_prefix+'_'+str(fold)+tt
                    # list_name = file_prefix + '_' + str(fold) + '.lst'
                    list_name = file_prefix + '.lst'
                    list_df = pd.read_csv(list_name, sep='\t', names=['id', 'label', 'file'], header=None)
                    new_list = ""
                    if self.rank == 'all-in-one':
                        taxon2id = dict()
                    for index, row in list_df.iterrows():
                        new_list = new_list + str(row['id'])
                        taxon = id2taxon[row['label']]
                        higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                        for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                            if item is np.nan:
                                item = 'nan'
                                # print('adding '+str(item))
                            if self.rank == 'all-in-one':
                                if item not in taxon2id:
                                    taxon2id[item] = len(taxon2id) + 1
                                new_list = new_list + '\t' + str(taxon2id[item])
                            elif self.rank == 'hierarchical':
                                if i not in taxon2id:
                                    taxon2id[i] = dict()
                                if item not in taxon2id[i]:
                                    taxon2id[i][item] = len(taxon2id[i]) + 1
                                new_list = new_list + '\t' + str(taxon2id[i][item])
                            if i not in self.classes:
                                self.classes[i] = set()
                            self.classes[i].add(item)
                        new_list = new_list + '\t' + row['file'] + '\n'
                        # fn = self.file_prefix + '_' + str(fold) + '.lst'
                        # print('##########')
                    # with open(fn, 'wt') as out_file:
                    with open(list_name, 'wt') as out_file:
                        out_file.write(new_list)

        elif self.file_prefix.split('_')[-2]=='xval':
            ## building list
            for fold in range(self.k):
                # load list as df
                list_name = self.file_prefix + '_' + str(fold) + '.lst'
                list_df = pd.read_csv(list_name, sep='\t', names=['id', 'label', 'file'], header=None)
                new_list = ""
                if self.rank == 'all-in-one':
                    taxon2id = dict()
                for index, row in list_df.iterrows():
                    new_list = new_list + str(row['id'])
                    taxon = id2taxon[row['label']]
                    higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                    for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                        if item is np.nan:
                            item = 'nan'
                            # print('adding '+str(item))
                        if self.rank == 'all-in-one':
                            if item not in taxon2id:
                                taxon2id[item] = len(taxon2id) + 1
                            new_list = new_list + '\t' + str(taxon2id[item])
                        elif self.rank == 'hierarchical':
                            if i not in taxon2id:
                                taxon2id[i] = dict()
                            if item not in taxon2id[i]:
                                taxon2id[i][item] = len(taxon2id[i]) + 1
                            new_list = new_list + '\t' + str(taxon2id[i][item])
                        if i not in self.classes:
                            self.classes[i] = set()
                        self.classes[i].add(item)
                    new_list = new_list + '\t' + row['file'] + '\n'
                    fn = self.file_prefix + '_' + str(fold) + '.lst'
                    # print('##########')
                with open(fn, 'wt') as out_file:
                    out_file.write(new_list)
                # ## building combined train list
                # fout = self.file_prefix + '_train_' + str(fold) + '.lst'
                # with open(fout, 'w') as outfile:
                #     for i in range(self.k):
                #         if i != fold:  # leave validation list out
                #             fin = self.file_prefix + '_' + str(i) + '.lst'
                #             with open(fin) as infile:
                #                 outfile.write(infile.read())

        if self.file_prefix.split('_')[-2]=='tt-split' and self.file_prefix.split('_')[-3]=='orig':
            train_list_name = self.file_prefix + '_train' + '.lst'
            train_list_df = pd.read_csv(train_list_name, sep='\t', names=['id', 'label', 'file'], header=None)
            new_list = ""
            for index, row in train_list_df.iterrows():
                new_list = new_list + str(row['id'])
                taxon = id2taxon[row['label']]
                higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                    if item is np.nan:
                        item = 'nan'
                        # print('adding '+str(item))
                    if self.rank == 'all-in-one':
                        if item not in taxon2id:
                            taxon2id[item] = len(taxon2id) + 1
                        new_list = new_list + '\t' + str(taxon2id[item])
                    elif self.rank == 'hierarchical':
                        if i not in taxon2id:
                            taxon2id[i] = dict()
                        if item not in taxon2id[i]:
                            taxon2id[i][item] = len(taxon2id[i]) + 1
                        new_list = new_list + '\t' + str(taxon2id[i][item])
                    if i not in self.classes:
                        self.classes[i] = set()
                    self.classes[i].add(item)
                new_list = new_list + '\t' + row['file'] + '\n'
                fn = self.file_prefix + '_train' + '.lst'
                # print('##########')
            with open(fn, 'wt') as out_file:
                out_file.write(new_list)
            # load test list as df
            test_list_name = self.file_prefix + '_test' + '.lst'
            test_list_df = pd.read_csv(test_list_name, sep='\t', names=['id', 'label', 'file'], header=None)
            new_list = ""
            for index, row in test_list_df.iterrows():
                new_list = new_list + str(row['id'])
                taxon = id2taxon[row['label']]
                higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                    if item is np.nan:
                        item = 'nan'
                        # print('adding '+str(item))
                    if self.rank == 'all-in-one':
                        if item not in taxon2id:
                            taxon2id[item] = len(taxon2id) + 1
                        new_list = new_list + '\t' + str(taxon2id[item])
                    elif self.rank == 'hierarchical':
                        if i not in taxon2id:
                            taxon2id[i] = dict()
                        if item not in taxon2id[i]:
                            taxon2id[i][item] = len(taxon2id[i]) + 1
                        new_list = new_list + '\t' + str(taxon2id[i][item])
                    if i not in self.classes:
                        self.classes[i] = set()
                    self.classes[i].add(item)
                new_list = new_list + '\t' + row['file'] + '\n'
                fn = self.file_prefix + '_test' + '.lst'
                # print('##########')
            with open(fn, 'wt') as out_file:
                out_file.write(new_list)

        if self.file_prefix.split('_')[-2]=='tt-split' and self.file_prefix.split('_')[-3]=='oversampled':
            # load train list as df
            list_name = self.file_prefix + '.lst'
            list_df = pd.read_csv(list_name, sep='\t', names=['id', 'label', 'file'], header=None)
            new_list = ""
            for index, row in list_df.iterrows():
                new_list = new_list + str(row['id'])
                taxon = id2taxon[row['label']]
                higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                    if item is np.nan:
                        item = 'nan'
                        # print('adding '+str(item))
                    if self.rank == 'all-in-one':
                        if item not in taxon2id:
                            taxon2id[item] = len(taxon2id) + 1
                        new_list = new_list + '\t' + str(taxon2id[item])
                    elif self.rank == 'hierarchical':
                        if i not in taxon2id:
                            taxon2id[i] = dict()
                        if item not in taxon2id[i]:
                            taxon2id[i][item] = len(taxon2id[i]) + 1
                        new_list = new_list + '\t' + str(taxon2id[i][item])
                    if i not in self.classes:
                        self.classes[i] = set()
                    self.classes[i].add(item)
                new_list = new_list + '\t' + row['file'] + '\n'
                fn = self.file_prefix + '.lst'
                # print('##########')
            with open(fn, 'wt') as out_file:
                out_file.write(new_list)
        for key in self.classes:
            self.classes[key] = len(self.classes[key])
        mapping_df = pd.DataFrame(taxon2id.items())
        return mapping_df

    def _create_ml_list(self):
        # create mapping dicts
        # id2taxon = mapping_df.to_dict()['taxon']
        id2taxon = self.global_mapping.to_dict()['taxon']
        missing_values = ['', 'unknown', 'unclassified', 'unidentified']
        taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        csv_path = os.path.join(self.root_path, 'im.merged.v10032020_unique_id_set.csv')
        df = pd.read_csv(csv_path, na_values=missing_values)[taxonomic_groups]

        if self.file_prefix.split('_')[-2] == 'xval' and self.file_prefix.split('_')[-3] == 'oversampled':
            mapping_part = dict()
            for tt in ['_train','_test']:
                for fold in range(self.k):
                    file_prefix = self.file_prefix+'_'+str(fold)+tt
                    # list_name = file_prefix + '_' + str(fold) + '.lst'
                    list_name = file_prefix + '.lst'
                    list_df = pd.read_csv(list_name, sep='\t', names=['id', 'label', 'file'], header=None)
                    new_list = ""
                    for index, row in list_df.iterrows():
                        new_list = new_list + str(row['id'])
                        taxon = id2taxon[row['label']]
                        higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                        for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                            global_label = self.global_mapping.loc[self.global_mapping['taxon']==item]['id'].values[0]
                            new_list = new_list + '\t' + str(global_label)
                        new_list = new_list + '\t' + row['file'] + '\n'
                    with open(list_name, 'wt') as out_file:
                        out_file.write(new_list)

        elif self.file_prefix.split('_')[-2]=='xval':
            ## building list
            for fold in range(self.k):
                # load list as df
                list_name = self.file_prefix + '_' + str(fold) + '.lst'
                list_df = pd.read_csv(list_name, sep='\t', names=['id', 'label', 'file'], header=None)
                new_list = ""
                for index, row in list_df.iterrows():
                    new_list = new_list + str(row['id'])
                    taxon = id2taxon[row['label']]
                    higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                    for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                        global_label = self.global_mapping.loc[self.global_mapping['taxon'] == item]['id'].values[0]
                        new_list = new_list + '\t' + str(global_label)
                    new_list = new_list + '\t' + row['file'] + '\n'
                    fn = self.file_prefix + '_' + str(fold) + '.lst'
                with open(fn, 'wt') as out_file:
                    out_file.write(new_list)


        if self.file_prefix.split('_')[-2]=='tt-split' and self.file_prefix.split('_')[-3]=='orig':
            train_list_name = self.file_prefix + '_train' + '.lst'
            train_list_df = pd.read_csv(train_list_name, sep='\t', names=['id', 'label', 'file'], header=None)
            new_list = ""
            for index, row in train_list_df.iterrows():
                new_list = new_list + str(row['id'])
                taxon = id2taxon[row['label']]
                higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                    global_label = self.global_mapping.loc[self.global_mapping['taxon'] == item]['id'].values[0]
                    new_list = new_list + '\t' + str(global_label)
                new_list = new_list + '\t' + row['file'] + '\n'
                fn = self.file_prefix + '_train' + '.lst'
            with open(fn, 'wt') as out_file:
                out_file.write(new_list)
            # load test list as df
            test_list_name = self.file_prefix + '_test' + '.lst'
            test_list_df = pd.read_csv(test_list_name, sep='\t', names=['id', 'label', 'file'], header=None)
            new_list = ""
            for index, row in test_list_df.iterrows():
                new_list = new_list + str(row['id'])
                taxon = id2taxon[row['label']]
                higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                    global_label = self.global_mapping.loc[self.global_mapping['taxon'] == item]['id'].values[0]
                    new_list = new_list + '\t' + str(global_label)
                new_list = new_list + '\t' + row['file'] + '\n'
                fn = self.file_prefix + '_test' + '.lst'
            with open(fn, 'wt') as out_file:
                out_file.write(new_list)

        if self.file_prefix.split('_')[-2]=='tt-split' and self.file_prefix.split('_')[-3]=='oversampled':
            # load train list as df
            list_name = self.file_prefix + '.lst'
            list_df = pd.read_csv(list_name, sep='\t', names=['id', 'label', 'file'], header=None)
            new_list = ""
            for index, row in list_df.iterrows():
                new_list = new_list + str(row['id'])
                taxon = id2taxon[row['label']]
                higher_taxons = df.loc[df['species'] == taxon].iloc[0, :]
                for i, item in enumerate(higher_taxons.to_list()):  # add additional labels
                    global_label = self.global_mapping.loc[self.global_mapping['taxon'] == item]['id'].values[0]
                    new_list = new_list + '\t' + str(global_label)
                new_list = new_list + '\t' + row['file'] + '\n'
                fn = self.file_prefix + '.lst'
            with open(fn, 'wt') as out_file:
                out_file.write(new_list)


    def _make_unique_global_mapping_lists(self, mapping_part, csv_path):
        global_taxon_col = []
        global_id_col = []
        # mapping = dict()
        missing_values = ['', 'unknown', 'unclassified', 'unidentified']
        taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        df = pd.read_csv(csv_path, na_values=missing_values)[taxonomic_groups]

        # for k in mapping_part:
        #     for index, row in mapping_part[k].iterrows():
        #         if row['taxon'] not in mapping:
        #             mapping[row['taxon']] = len(mapping)

        for i in range(self.k):
            # open list file
            for tt in ['_train','_test']:
                list_name = self.file_prefix+'_'+str(i)+tt+'.lst'
                list_df = pd.read_csv(list_name, sep='\t', names=['id', 'label', 'file'], header=None)
                new_list = ""
                for index, row in list_df.iterrows():# process list file line by line
                    new_list = new_list + str(row['id'])
                    label = row['label']
                    taxon = mapping_part[i]['taxon'].loc[mapping_part[i]['id']==label].values[0]
                    ###################################
                    global_label = self.global_mapping.loc[self.global_mapping['taxon'] == taxon]['id'].values[0]
                    global_taxon_col.append(taxon)
                    global_id_col.append(global_label)
                    new_list = new_list + '\t' + str(global_label)
                    ###################################
                    # new_list = new_list + '\t' + str(mapping[taxon])
                    abs_path = os.path.join(self.root_path, self.rank, 'x%s%s' %(i,tt), row['file'])
                    new_list = new_list + '\t' + abs_path + '\n'
                # print('##########')
                with open(list_name, 'wt') as out_file:
                    out_file.write(new_list)
        m_df = pd.DataFrame()
        # m_df['taxon'] = mapping.keys()
        # m_df['id'] = mapping.values()
        m_df['taxon'] = global_taxon_col
        m_df['id'] = global_id_col
        return m_df

    def _transform_to_global_mapping(self, mapping_df, list_path):
        global_taxon_col = []
        global_id_col = []

        id2taxon = mapping_df.to_dict()['taxon']
        list_df = pd.read_csv(list_path, sep='\t', names=['id', 'label', 'file'], header=None)

        new_list = ""
        for index, row in list_df.iterrows():# process list file line by line
            new_list = new_list + str(row['id'])
            taxon = id2taxon[row['label']]
            global_label = self.global_mapping.loc[self.global_mapping['taxon'] == taxon]['id'].values[0]
            global_taxon_col.append(taxon)
            global_id_col.append(global_label)
            new_list = new_list + '\t' + str(global_label)
            new_list = new_list + '\t' + row['file'] + '\n'
        # print('##########')
        shutil.copy(list_path, list_path[:-4]+'_no_global_mapping.lst')
        with open(list_path, 'wt') as out_file:
            out_file.write(new_list)

        m_df = pd.DataFrame()
        m_df['taxon'] = global_taxon_col
        m_df['id'] = global_id_col
        return m_df

    def _build_global_mapping(self, csv_path):
        # try to load global mapping csv, if not exist build
        if os.path.exists(os.path.join(self.root_path, 'global_mapping.csv')):
            print('loading global mapping')
            self.global_mapping = pd.read_csv(os.path.join(self.root_path, 'global_mapping.csv'))
        else:
            print('creating global mapping')
            taxon_col = []
            id_col = []
            missing_values = ['', 'unknown', 'unclassified', 'unidentified']
            taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
            df = pd.read_csv(csv_path, na_values=missing_values)[taxonomic_groups]

            p = set(df['phylum'].values)
            c = set(df['class'].values)
            o = set(df['order'].values)
            f = set(df['family'].values)
            g = set(df['genus'].values)
            s = set(df['species'].values)
            taxon_col.extend(list(p))
            taxon_col.extend(list(c))
            taxon_col.extend(list(o))
            taxon_col.extend(list(f))
            taxon_col.extend(list(g))
            taxon_col.extend(list(s))
            id_col = list(range(len(taxon_col)))

            self.global_mapping = pd.DataFrame()
            self.global_mapping['taxon'] = taxon_col
            self.global_mapping['id'] = id_col

            self.global_mapping.to_csv(os.path.join(self.root_path, 'global_mapping.csv'))


    def _create_recordIO(self):
        # creates lists for and RecordIO files
        # according to xval, orig_tt-split or oversampled_tt-split

        csv_path = os.path.join(self.root_path, 'im.merged.v10032020_unique_id_set.csv')
        # 1) set up one global unique mapping df, containing one col with all taxons and one with all ids -> self._get_global_mapping
        # self._build_global_mapping(csv_path)
        # 2) use this unique global mapping on ALL lists -> self._make_unique_mapping_lists

        # create lists
        if self.file_prefix.split('_')[-2] == 'xval' and self.file_prefix.split('_')[-3] == 'oversampled':
            mapping_part = dict()
            for tt in ['_train','_test']:
                for i in range(self.k):
                    rank_path = os.path.join(self.root_path, self.rank, 'x%s%s' %(i,tt))
                    file_prefix = self.file_prefix+'_'+str(i)+tt
                    i2r = Im2Rec([file_prefix, rank_path, '--recursive', '--list', '--pack-label', '--num-thread', str(self.num_workers)])
                    raw_data = StringIO(i2r.str_mapping)
                    mapping_part[i] = pd.read_csv(raw_data, sep=' ', names=['taxon', 'id'], header=None)
            mapping_df = self._make_unique_global_mapping_lists(mapping_part, csv_path)
        elif self.file_prefix.split('_')[-2] == 'xval':
            i2r = Im2Rec([self.file_prefix, self.rank_path, '--recursive', '--list', '--pack-label', '--chunks', str(self.k),'--num-thread', str(self.num_workers)])
            raw_data = StringIO(i2r.str_mapping)
            mapping = pd.read_csv(raw_data, sep=' ', names=['taxon', 'id'], header=None)
            for fold in range(self.k):
                list_path = os.path.join(self.rank_path, self.file_prefix + '_' + str(fold) + '.lst')
                mapping_new = self._transform_to_global_mapping(mapping, list_path)
            mapping = mapping_new
        elif self.file_prefix.split('_')[-2] == 'tt-split' and self.file_prefix.split('_')[-3] == 'orig':
            # i2r = Im2Rec([self.file_prefix, self.rank_path, '--recursive', '--list', '--pack-label', '--test-ratio', str(self.test_ratio), '--train-ratio', str(self.train_ratio),'--num-thread', str(self.num_workers)], self.global_mapping)
            i2r = Im2Rec([self.file_prefix, self.rank_path, '--recursive', '--list', '--pack-label', '--test-ratio', str(self.test_ratio), '--train-ratio', str(self.train_ratio), '--num-thread',str(self.num_workers)])
            raw_data = StringIO(i2r.str_mapping)
            mapping = pd.read_csv(raw_data, sep=' ', names=['taxon', 'id'], header=None)
            list_path = os.path.join(self.rank_path, self.file_prefix + '_train' + '.lst')
            self._transform_to_global_mapping(mapping, list_path)
            list_path = os.path.join(self.rank_path, self.file_prefix + '_test' + '.lst')
            self._transform_to_global_mapping(mapping, list_path)
            mapping.to_csv(self.file_prefix+'.mapping')
        elif self.file_prefix.split('_')[-2] == 'tt-split' and self.file_prefix.split('_')[-3] == 'oversampled':
            i2r = Im2Rec([self.file_prefix, self.rank_path, '--recursive', '--list', '--pack-label', '--num-thread', str(self.num_workers)])
            raw_data = StringIO(i2r.str_mapping)
            mapping = pd.read_csv(raw_data, sep=' ', names=['taxon', 'id'], header=None)
            list_path = os.path.join(self.rank_path, self.file_prefix + '.lst')
            mapping = self._transform_to_global_mapping(mapping, list_path)

        # if not(self.file_prefix.split('_')[-2] == 'xval' and self.file_prefix.split('_')[-3] == 'oversampled'):
        #     # get the mapping
        #     raw_data = StringIO(i2r.str_mapping)
        #     mapping_df = pd.read_csv(raw_data, sep=' ', names=['taxon', 'id'], header=None)

        if self.rank == 'all-in-one' or self.rank == 'hierarchical' : # add multi-labels
            self._create_ml_list()

        # create RecordIO
        if self.rank == 'hierarchical' and  self.file_prefix.split('_')[-2] == 'xval':
            for fold in range(self.k):
                # x7-approach
                # if self.file_prefix.split('_')[-2] == 'xval':
                if self.file_prefix.split('_')[-3] == 'oversampled':
                    fout = self.file_prefix + '_' + str(fold) +'_train.lst'
                    list_arg = self.file_prefix + '_' + str(fold) + '_test' + '.lst'
                else:
                    fout = self.file_prefix + '_train_' + str(fold) + '.lst'
                    list_arg = self.file_prefix + '_' + str(fold) + '.lst'
                print('creating ' + fout)
                print('creating ' + list_arg)
                i2r = Im2Rec([fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
                i2r = Im2Rec([list_arg, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
                # elif self.file_prefix.split('_')[-2] == 'tt-split' and self.file_prefix.split('_')[-3] == 'oversampled':
                #     fout = self.file_prefix + '.lst'
                #     print('creating ' + fout)
                #     i2r = Im2Rec(
                #         [fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
                # else:
                #     fout = self.file_prefix + '_train' + '.lst'
                #     print('creating ' + fout)
                #     list_arg = self.file_prefix + '_test' + '.lst'
                #     print('creating ' + list_arg)
                #     i2r = Im2Rec(
                #         [fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
                #     i2r = Im2Rec([list_arg, self.rank_path, '--recursive', '--pass-through', '--num-thread',
                #                   str(self.num_workers)])

        elif self.file_prefix.split('_')[-2] == 'xval':
            for fold in range(self.k):
                if self.file_prefix.split('_')[-3] == 'oversampled':
                    fout = self.file_prefix + '_ensembled_train_' + str(fold) + '.lst'
                else:
                    fout = self.file_prefix + '_train_' + str(fold) + '.lst'
                with open(fout, 'w') as outfile:
                    for i in range(self.k):
                        if i != fold:  # leave validation list out
                            if self.file_prefix.split('_')[-3] == 'oversampled':
                                fin = self.file_prefix + '_' + str(i) +'_train'+ '.lst'
                            else:
                                fin = self.file_prefix + '_' + str(i) + '.lst'
                            with open(fin) as infile:
                                outfile.write(infile.read())
                # create record for train and test
                if self.file_prefix.split('_')[-3] == 'oversampled':
                    fout = self.file_prefix + '_ensembled_train_' + str(fold) + '.lst'
                    list_arg = self.file_prefix + '_' + str(fold) +'_test'+ '.lst'
                else:
                    fout = self.file_prefix + '_train_' + str(fold) + '.lst'
                    list_arg = self.file_prefix + '_' + str(fold) + '.lst'
                print('creating ' + fout)
                print('creating ' + list_arg)
                i2r = Im2Rec([fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
                i2r = Im2Rec([list_arg, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
        if self.file_prefix.split('_')[-2] == 'tt-split' and self.file_prefix.split('_')[-3] == 'orig':
            fout = self.file_prefix + '_train' + '.lst'
            print('creating ' + fout)
            list_arg = self.file_prefix + '_test' + '.lst'
            print('creating ' + list_arg)
            # i2r = Im2Rec([fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)], self.global_mapping)
            # i2r = Im2Rec([list_arg, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)], self.global_mapping)
            i2r = Im2Rec([fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
            i2r = Im2Rec([list_arg, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
        if self.file_prefix.split('_')[-2] == 'tt-split' and self.file_prefix.split('_')[-3] == 'oversampled':
            fout = self.file_prefix + '.lst'
            print('creating ' + fout)
            i2r = Im2Rec([fout, self.rank_path, '--recursive', '--pass-through', '--num-thread', str(self.num_workers)])
        # self.global_mapping.to_csv(os.path.join(self.rank_path,'mapping.csv'))

    def load_rec(self, fold=0):
        label_width = 1
        shuffle_test = True
        if self.rank == 'all-in-one' or self.rank == 'hierarchical':
            label_width = 6

        if self.rank == 'hierarchical' and self.file_prefix.split('_')[-2] == 'xval':
                if self.file_prefix.split('_')[-3] == 'oversampled':
                    file_prefix = self.file_prefix + '_' + str(fold) + '_train'
                    train_rec_path = file_prefix + '.rec'
                    train_idx_path = file_prefix + '.idx'
                    train_lst_path = file_prefix + '.lst'
                    file_prefix = self.file_prefix + '_' + str(self.k-1) + '_test'
                    test_rec_path = file_prefix + '.rec'
                    test_idx_path = file_prefix + '.idx'
                    test_lst_path = file_prefix + '.lst'
                else:
                    file_prefix = self.file_prefix + '_train_' + str(fold)
                    train_rec_path = file_prefix + '.rec'
                    train_idx_path = file_prefix + '.idx'
                    train_lst_path = file_prefix + '.lst'
                    file_prefix = self.file_prefix + '_' + str(self.k-1)
                    test_rec_path = file_prefix + '.rec'
                    test_idx_path = file_prefix + '.idx'
                    test_lst_path = file_prefix + '.lst'
            # else:
            #     shuffle_test = False
            #     train_rec_path = self.file_prefix + '_train' + '.rec'
            #     train_idx_path = self.file_prefix + '_train' + '.idx'
            #     train_lst_path = self.file_prefix + '_train' + '.lst'
            #     test_rec_path = self.file_prefix + '_test' + '.rec'
            #     test_idx_path = self.file_prefix + '_test' + '.idx'
            #     test_lst_path = self.file_prefix + '_test' + '.lst'
        elif self.mode.split('_')[-2]=='tt-split' and self.mode.split('_')[-3]=='oversampled':
            shuffle_test = False
            # creating train separately
            rank_path = os.path.join(self.root_path, self.rank, 'train')
            file_prefix = rank_path + '/' + self.rank + '_train'  + self.mode
            train_rec_path = file_prefix + '.rec'
            train_idx_path = file_prefix + '.idx'
            train_lst_path = file_prefix + '.lst'
            rank_path = os.path.join(self.root_path, self.rank, 'test')
            file_prefix = rank_path + '/' + self.rank + '_test'  + self.mode
            test_rec_path = file_prefix + '.rec'
            test_idx_path = file_prefix + '.idx'
            test_lst_path = file_prefix + '.lst'
        elif self.mode.split('_')[-2]=='xval' and self.mode.split('_')[-3]=='oversampled':
            train_rec_path = self.file_prefix + '_ensembled_train_' + str(fold) + '.rec'
            train_idx_path = self.file_prefix + '_ensembled_train_' + str(fold) + '.idx'
            train_lst_path = self.file_prefix + '_ensembled_train_' + str(fold) + '.lst'
            test_rec_path = self.file_prefix + '_' + str(fold) + '_test' + '.rec'
            test_idx_path = self.file_prefix + '_' + str(fold) + '_test' + '.idx'
            test_lst_path = self.file_prefix + '_' + str(fold) + '_test' + '.lst'
        elif self.file_prefix.split('_')[-2] == 'xval':
            train_rec_path = self.file_prefix + '_train_' + str(fold) + '.rec'
            train_idx_path = self.file_prefix + '_train_' + str(fold) + '.idx'
            train_lst_path = self.file_prefix + '_train_' + str(fold) + '.lst'
            test_rec_path = self.file_prefix + '_' + str(fold) + '.rec'
            test_idx_path = self.file_prefix + '_' + str(fold) + '.idx'
            test_lst_path = self.file_prefix + '_' + str(fold) + '.lst'
        else:
            shuffle_test = False
            train_rec_path = self.file_prefix + '_train' + '.rec'
            train_idx_path = self.file_prefix + '_train' + '.idx'
            train_lst_path = self.file_prefix + '_train' + '.lst'
            test_rec_path = self.file_prefix + '_test' + '.rec'
            test_idx_path = self.file_prefix + '_test' + '.idx'
            test_lst_path = self.file_prefix + '_test' + '.lst'

        self.train_data = ImageRecordIter(
            path_imgrec=train_rec_path,
            path_imgidx=train_idx_path,
            path_imglist=train_lst_path,
            aug_list=mx.image.CreateAugmenter((3, 224, 224), inter_method=1),
            data_shape=(3, 224, 224),
            # data_shape=(3, 356, 356),
            batch_size=self.batch_size,
            # random_resized_crop=True,
            # max_crop_size=356,
            # min_crop_size=356,
            resize=224,
            label_width=label_width,
            # inter_method=0,
            shuffle=True  # train true, test no
        )
        self.test_data = ImageRecordIter(
            path_imgrec=test_rec_path,
            path_imgidx=test_idx_path,
            path_imglist=test_lst_path,
            aug_list=mx.image.CreateAugmenter((3, 224, 224), inter_method=1),
            data_shape=(3, 224, 224),
            batch_size=self.batch_size,
            resize=224,
            label_width=label_width,
            shuffle=shuffle_test  # train true, test no
        )


    def transform(self, jitter_param = 0.4, lighting_param = 0.1, mode = 'train'):
        if mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomFlipLeftRight(),
                # transforms.RandomFlipTopBottom(),
                # transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                #                              saturation=jitter_param),
                # transforms.RandomLighting(lighting_param),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif mode == 'transformOversampled':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.RandomResizedCrop((224,224)),
                # transforms.Resize((224, 224)),
                transforms.RandomFlipLeftRight(),
                transforms.RandomFlipTopBottom(),
                transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                             saturation=jitter_param),
                transforms.RandomLighting(lighting_param)
            ])
        elif mode == 'naiveOversampled':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomResizedCrop((224,224)),
                transforms.Resize((224, 224)),
                transforms.RandomFlipLeftRight(),
                transforms.RandomFlipTopBottom(),
                # transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                #                              saturation=jitter_param),
                transforms.RandomLighting(lighting_param)
            ])
        return transform
