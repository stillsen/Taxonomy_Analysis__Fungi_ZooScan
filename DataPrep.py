import os, argparse, shutil, random, time, cv2
import pandas as pd
from gluoncv.utils import makedirs
import mxnet.image as img
import mxnet as mx
import ast
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as plti


class DataPrep:
    def __init__(self, path, rank, dataset, df, taxonomic_groups = None, multilabel_lvl=1):
        self.taxonomic_groups = taxonomic_groups
        self.imagefolder_path = ''

        # if multilabel_lvl == 1:
        #     cuts_path = os.path.join(path, 'cuts')
        #     makedirs(cuts_path)
        #     self.clean_up(path)
        # elif multilabel_lvl == 2:
        #     ml2_path = os.path.join(path, 'ml2')
        #     makedirs(ml2_path)
        #     path = ml2_path
        #     cuts_path = os.path.join(path, 'cuts')
        #     makedirs(cuts_path)

        if dataset == 'fun':
            # removes path/train, path/test and path/val
            # builds up mapping from unique identiefier to file
            # if enabled extracts circles and saves the cuts in self.cut_fun_images()
            # build imagefolders according to taxonomic rank
            # copies those folders/files according to sample ratio in path/test, path/train and path/val
            cuts_path = self.cut_fun_images(path)
            ids_dict = self.create_ids_dict(path=path)
            self.imagefolder_path = self.loadcreate_image_folders(path=cuts_path, rank=rank, df=df, dataset=dataset, ids_dict=ids_dict, multilabel_lvl=multilabel_lvl)
            self.sample(self.imagefolder_path)
        if dataset == 'zoo':
            self.imagefolder_path = self.loadcreate_image_folders(path=path, rank=rank, df=df, dataset=dataset, multilabel_lvl=multilabel_lvl)
            self.sample(self.imagefolder_path)
            pass


    def create_ids_dict(self, path):
        # create dictonary -> {mdate_skey : filename}
        print('creating unique file mapping')
        ids_dict = {}
        filenames = os.listdir(path)
        for file in filenames:
            file_no_ex = file[:-4]
            # rule out unwanted files such as Mo013.tif (not labeled) or  Eka_9.12.17_F(2).tif (pedridish flipped)
            if len(file_no_ex.split('_')) > 1:
                if not ((file_no_ex[0] == 'E' and file_no_ex.split('_')[-1][-1] == ')')):
                    # print('cutting', file)
                    skey = file_no_ex.split('_')[-1][0]
                    mdate = time.gmtime(os.path.getmtime(os.path.join(path, file)))
                    dd = str(mdate[2])
                    if len(str(mdate[1])) == 1:
                        mm = '0' + str(mdate[1])
                    else:
                        mm = str(mdate[1])
                    yy = str(mdate[0])[-2:]
                    mdate_skey = dd + '.' + mm + '.' + yy + '_' + skey
                    # print(file, mdate[2], mdate[1], mdate[0], skey, mdate_skey)
                    ids_dict[mdate_skey] = file
        return ids_dict

    def cut_fun_images(self, path):
        ##### has unresolved redundancy with self.create_ids_dict()
        # cuts those big fungy images and renames them uniquely and stores each cut sub-image into folder
        # path/cuts/
        # defines and declares
        # self.ids_dict
        # which is a dictionary to map file unique identifier (mixture of date and name and csv) to a unique filename


        resize = True
        resize_w = 356
        resize_h = 356

        cuts_path = os.path.join(path, 'cuts')
        if not os.path.exists(cuts_path):
            print('cutting images')
            makedirs(cuts_path)

            # 1)create dictonary -> {mdate_skey : filename}
            # 2)identify circles in group image, save circle_group_image and produce cuts respectivly
            # each image contains 12 image crops of fungis
            # these group images are uniquely identified via date of modification and sample key (skey = A, B, C, ...)
            # batch_no is later used to get the crop
            self.ids_dict = {}
            # for folderName, subfolders, filenames in os.walk(path):
            filenames = os.listdir(path)
            for file in filenames:
                file_no_ex = file[:-4]
                # rule out unwanted files such as Mo013.tif (not labeled) or  Eka_9.12.17_F(2).tif (pedridish flipped)
                if len(file_no_ex.split('_')) > 1:
                    if not ((file_no_ex[0] == 'E' and file_no_ex.split('_')[-1][-1] == ')')):
                        # print('cutting', file)
                        skey = file_no_ex.split('_')[-1][0]
                        mdate = time.gmtime(os.path.getmtime(os.path.join(path, file)))
                        dd = str(mdate[2])
                        if len(str(mdate[1])) == 1:
                            mm = '0' + str(mdate[1])
                        else:
                            mm = str(mdate[1])
                        yy = str(mdate[0])[-2:]
                        mdate_skey = dd + '.' + mm + '.' + yy + '_' + skey
                        # print(file, mdate[2], mdate[1], mdate[0], skey, mdate_skey)
                        self.ids_dict[mdate_skey] = file

                        ### identify petri dishes
                        if file[-3:] == 'tif':
                            self.cut_image(path=path, file=file, cuts_path=cuts_path, mdate_skey=mdate_skey, resize=resize, resize_w=resize_w,
                                           resize_h=resize_h)
        return cuts_path

    def loadcreate_image_folders(self, path, rank, df, dataset, ids_dict=None, multilabel_lvl=1):
        # dataset: fun or zoo
        # taxa: all or specific taxonomic rank
        # ids_dict is only used in fungi ds
        #
        # create subfolder for each isotope in cuts
        # path/cuts/isotope

        return_path = ''

        print('creating folder structure for image folder data sets')
        if dataset == 'fun':
            if multilabel_lvl == 1:
                print('multilabel lvl 1, fungi ds')
                # if not exists, create image folder
                rank_path = os.path.join(path, rank)
                if not os.path.exists(rank_path):
                    makedirs(rank_path)
                    # cuts_path = os.path.join(rank_path, 'cuts')
                    # makedirs(cuts_path)
                    ### get labels, identify images and crop, save image crop and label to folder
                    for index, row in df.iterrows():
                        # get mdate_skey (unique group image identifier) and label
                        mdate_skey = row['Scan.date']
                        pos = row['Pos.scan']
                        label = row[rank]
                        # get file
                        file_name = ids_dict[mdate_skey].rsplit('.', 1)[:-1][0] + '__' + mdate_skey + '__cut__' + str(pos - 1) + '.png'
                        file = os.path.join(path, file_name)
                        # create subfolder in cuts according to taxonomic resolution
                        this_taxon_to_folder = os.path.join(rank_path, str(label))
                        makedirs(this_taxon_to_folder)
                        # if the isomorph is classified in this taxonomic resolution, copy it to it's folder
                        if not pd.isnull(row[rank]):
                            if not os.path.exists(os.path.join(this_taxon_to_folder, file_name)):
                                shutil.copy(file, this_taxon_to_folder)
                return_path = rank_path

            elif multilabel_lvl == 2:
                # get labels, identify images
                # copy this isomorphe image to
                # each folder of it's classified taxonomic rank
                print('multilabel lvl 2, fungi ds')
                ml2_path = os.path.join(path, 'ml2')
                if not os.path.exists(ml2_path):
                    makedirs(ml2_path)
                    for index, row in df.iterrows():
                        # get mdate_skey (unique group image identifier) and label
                        mdate_skey = row['Scan.date']
                        pos = row['Pos.scan']
                        # get file
                        file_name = ids_dict[mdate_skey].rsplit('.', 1)[:-1][0] + '__' + mdate_skey + '__cut__' + str(pos - 1) + '.png'
                        file_from = os.path.join(path, file_name)

                        # loop over the taxonomic groups listed self.taxonomic_groups
                        # for each taxonomic group create a subfolder with the name of the current taxon according to the
                        # current taxonomic resolution
                        # copy the file there
                        for tg in self.taxonomic_groups:
                            if not pd.isnull(row[tg]):
                                taxon = row[tg]
                                # create a subfolder cutspath/m_lvl, in which all the taxon folder are to be created
                                # m_lvl_path = os.path.join(cuts_path, "m_lvl")
                                # if not os.path.exists(m_lvl_path): makedirs(m_lvl_path)
                                #
                                taxon_path = os.path.join(ml2_path, str(taxon))
                                if not os.path.exists(taxon_path): makedirs(taxon_path)

                                file_to = os.path.join(taxon_path, file_name)
                                # print("copying from:  %s to:  %s" %(file_from,file_to))
                                if not os.path.exists(file_to): shutil.copy(file_from, file_to)
                return_path = ml2_path

        elif dataset == 'zoo':
            if multilabel_lvl == 1:
                print('multilabel lvl 1, zooscan ds')
                # if not exists, create image folder
                rank_path = os.path.join(path, rank)
                if not os.path.exists(rank_path):
                    makedirs(rank_path)
                    # get all taxons in this rank
                    taxons = df.loc[df['rank'] == rank].taxon.unique()
                    # for each taxon
                    # copy its image folder to path\cuts\this_taxon_folder
                    for this_taxon in taxons:
                        this_taxon_from_folder = os.path.join(path, this_taxon)
                        this_taxon_to_folder = os.path.join(rank_path, this_taxon)
                        if not os.path.exists(this_taxon_to_folder):
                            # makedirs(this_taxon_to_folder)
                            shutil.copytree(this_taxon_from_folder, this_taxon_to_folder)
                return_path = rank_path

            elif multilabel_lvl == 2:
                print('multilabel lvl 2, zooscan ds')
                ml2_path = os.path.join(path, 'ml2')
                if not os.path.exists(ml2_path):
                    makedirs(ml2_path)
                    for index, row in df.iterrows():
                        file_name = str(row['objid']) + '.jpg'
                        taxon_path_from = os.path.join(path, row['taxon'])
                        file_from = os.path.join(taxon_path_from, file_name)

                        # loop over the taxonomic groups listed self.taxonomic_groups
                        # for each taxonomic group create a subfolder with the name of the current taxon according to the
                        # current taxonomic resolution
                        # copy the file there
                        ranks = ast.literal_eval(row['ranks'])
                        # !!! I could also loop over lineage directly, BUT I wouldn't know the rank or also include taxons I do not know the rank of
                        # starting in the back
                        for i, tg in enumerate(ranks[::-1]):
                            if tg != '' and tg != 'no rank':
                                # ast.literal_eval pasrses the string representation of a list and translates it into a list
                                taxon = ast.literal_eval(row['lineage'])[-(i+1)]

                                taxon_path_to = os.path.join(ml2_path, str(taxon))
                                if not os.path.exists(taxon_path_to): makedirs(taxon_path_to)

                                file_to = os.path.join(taxon_path_to, file_name)
                                # print("copying from:  %s to:  %s" %(file_from,file_to))
                                if not os.path.exists(file_to): shutil.copy(file_from, file_to)
                return_path = ml2_path
        return return_path

    def sample(self, path, ratio = {'train':0.85, 'test': 0.1, 'validation': 0.05}):
        # Create directories:
        # already exists: path/cuts/
        # path/cuts/train/
        # path/cuts/test/
        # path/cuts/val/
        # sample images in

        print('sampling into directories: train, test, val')
        train_path = os.path.join(path, 'train')
        val_path = os.path.join(path, 'val')
        test_path = os.path.join(path, 'test')

        if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
            # creating train, validation and test set, 85:5:10
            # all subdirs in cut_path without files
            subdirs = [os.path.join(path, x) for x in os.listdir(path) if
                       not os.path.isfile(os.path.join(path, x))]
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
            train_images = random.sample(filenames, round(len(filenames) * .8))
            filenames = [elem for elem in filenames if elem not in train_images]
            val_images = random.sample(filenames, round(len(filenames) * .05))
            filenames = [elem for elem in filenames if elem not in val_images]
            test_images = filenames

            makedirs(train_path)
            makedirs(val_path)
            makedirs(test_path)

            for l in subdirs_names:
                makedirs(os.path.join(train_path, l))
                makedirs(os.path.join(val_path, l))
                makedirs(os.path.join(test_path, l))

            # Copy files to corresponding directory
            for file in train_images:
                split_list = file.split('/')
                label = split_list[-2]
                to_path = os.path.join(train_path, label)
                shutil.copy(file, to_path)

            for file in val_images:
                split_list = file.split('/')
                label = split_list[-2]
                to_path = os.path.join(val_path, label)
                shutil.copy(file, to_path)

            for file in test_images:
                split_list = file.split('/')
                label = split_list[-2]
                to_path = os.path.join(test_path, label)
                shutil.copy(file, to_path)

    def render_as_image(self, a):
        img = a.asnumpy() # convert to numpy array
        # img = img.transpose((1, 2, 0))  # Move channel to the last dimension
        img = img.astype(np.uint8)  # use uint8 (0-255)
        plt.imshow(img)
        plt.show()

    def sort_circles(self, circles, limit):
        circles = sorted(circles, key=lambda x: x[1])
        old_y = circles[0][1]
        for (i,(x,y,r)) in enumerate(circles):
            if abs(old_y - y) > limit:
                old_y = circles[i][1]
            circles[i] = np.append(circles[i], old_y)

        circles = sorted(circles, key=lambda x: (x[3], x[0]))
        circles = [(x,y,r) for (x,y,r,z) in circles]
        return circles

    def cut_image(self, path, file, cuts_path, mdate_skey, resize_w=None, resize_h=None, resize=False):

        #check if already computed
        output_file_name = file.rsplit('.', 1)[:-1][0] + '__' + mdate_skey + '__cuts' + '.png'
        output_file = os.path.join(cuts_path, output_file_name)
        if not os.path.exists(output_file):
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path)
            image = cv2.resize(image, dsize=(2550, 3509), interpolation=cv2.INTER_AREA)
            image_gs = cv2.imread(img_path, 0)
            image_gs = cv2.resize(image_gs, dsize=(2550, 3509), interpolation=cv2.INTER_AREA)
            # Create mask for cutting crop
            height, width = image_gs.shape
            mask = np.zeros((height, width), np.uint8)
            # create parameters according to image resolution
            # since resize only one set needed
            dp = 1.4
            minDist = 650
            param1 = 10
            param2 = 100
            minRadius = 250  # HoughCircles will look for circles at minimum this size
            maxRadius = 350

            # identify position of petridishes
            limit_difference = minRadius//2
            output = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray,
                                       cv2.HOUGH_GRADIENT,
                                       dp=dp,
                                       minDist=minDist,  # number of pixels center of circles should be from each other, hardcode
                                       param1=param1,
                                       param2=param2,
                                       minRadius=minRadius,  # HoughCircles will look for circles at minimum this size
                                       maxRadius=maxRadius
                                       )
            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")

                # no need to sort, if cutting image as whole, not truhue! i neeeeeeeeeeeed x_x
                circles = self.sort_circles(circles, limit=limit_difference)

                # loop over the (x, y) coordinates and radius of the circles

                for (i,(x, y, r)) in enumerate(circles):
                    # print(i, ' circle: ', y, x)

                    ## create group cut image
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
                    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    cv2.putText(output,
                                text=str(i),
                                org=(x, y),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(255, 255, 255),
                                lineType=2)
                    ## cut crop and save to folder
                    # Copy that image using that mask
                    masked_data = cv2.bitwise_and(image, image, mask=mask)

                    # Apply Threshold
                    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

                    # Find Contour
                    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    (x, y, w, h) = cv2.boundingRect(contours[0][0])

                    # Crop masked_data
                    crop = masked_data[y:y + h, x:x + w]
                    if resize:
                        crop = cv2.resize(crop, dsize=(resize_w, resize_h), interpolation=cv2.INTER_AREA)
                    # save single cut image to cuts path
                    output_file_name = file.rsplit('.', 1)[:-1][0] + '__' + mdate_skey + '__cut__' + str(i) + '.png'
                    output_file = os.path.join(cuts_path, output_file_name)
                    cv2.imwrite(output_file, crop)



                # save group cut image for validation
                output_file_name = file.rsplit('.', 1)[:-1][0] + '__' + mdate_skey + '__cuts' + '.png'
                output_file = os.path.join(cuts_path, output_file_name)
                cv2.imwrite(output_file, output)

    def clean_up(self, path):
        # removes directories, as they need to be made anew for each taxonomic rank
        # removed are:
        # path/train
        # path/val
        # path/test
        print('cleaning directories')
        cuts_path = os.path.join(path, 'cuts')
        train_path = os.path.join(path, 'train')
        val_path = os.path.join(path, 'val')
        test_path = os.path.join(path, 'test')

        if os.path.isdir(train_path):
            shutil.rmtree(train_path)
        if os.path.isdir(test_path):
            shutil.rmtree(test_path)
        if os.path.isdir(val_path):
            shutil.rmtree(val_path)
        # only erase subfolders, not files
        if os.path.isdir(cuts_path):
            for item in os.listdir(cuts_path):
                subdir = os.path.join(cuts_path,item)
                if os.path.isdir(subdir):
                    shutil.rmtree(subdir)

