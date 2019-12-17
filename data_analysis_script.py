# Analysis script: gives...
# taxonomic class distribution
# taxonomic distribution with highlighted  class
# taxonomic distribution within a class
# for fungi ds and zooscan sd

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from ete3 import NCBITaxa
import pickle
import os.path
from os import path

def load_ncbi():
    # file_name = '/home/stillsen/Documents/Uni/MasterArbeit/TaxonomicImageClassifier/ncbi_db.pickle'
    #
    # if not path.exists(file_name):
    #     print('creating local ncbi db ...')
    #     ncbi = NCBITaxa()
    #     ncbi.update_taxonomy_database()
    #     pickle.dump(ncbi, open(file_name, 'wb'))
    # else:
    #     print('loading local ncbi db ...')
    #     ncbi = pickle.load(open(file_name,'rb'))
    # print('ncbi loaded')
    ncbi = NCBITaxa()
    ncbi.update_taxonomy_database()
    return ncbi

def tax_distr(df):
    return []

def rank_distr(df):
    return []

def prepare_fun_df(df, higher_taxonomic_groups = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']):
    # add a taxon and maximum classification class (rank) column
    #taxonomic_groups = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    #creating a new column for maximum taxonomic classification
    # initiate with species level
    df['taxon'] = df['species']
    df['rank'] = np.nan
    #replace nans
    # every nan in species level is replace by the class one level higher, recursively for higher orders
    # additionally set the maximum classification class(rank) for df['rank']

    for index, row in df.iterrows():
        rank = 'species'
        for higher_taxa in reversed(higher_taxonomic_groups):
            if pd.isnull(df['taxon'][index]):
                df['taxon'][index] = df[higher_taxa][index]
                rank = higher_taxa
            #set the maximum classification class
            df['rank'][index] = rank
            if not pd.isnull(df['taxon'][index]):
                break
    return df

def prepare_zoo_df(csv_path, na_values):
    df = pd.read_csv(csv_path, na_values=na_values)
    file = '/home/stillsen/Documents/Uni/MasterArbeit/Source/Taxonomy_Analysis__Fungi_ZooScan/zoo_df.csv'
    if not path.exists(file):
        # load ncbi database
        ncbi = load_ncbi()
        print('df size before clean %s'%(len(df)))
        #add all morphotypes to na and drop them, as atm I don't know how to handle them
        prev_taxon = ''
        for index, row in df.iterrows():
            # print('cleaning row %s' %index)
            taxon = row['taxon']
            #is already nan
            if pd.isna(taxon):
                continue
            if prev_taxon != taxon:
                prev_taxon = taxon
                # some weird value which isn't string
                if not isinstance(taxon, str):
                    na_values.append(taxon)
                    print('type: %s'%taxon)
                    continue
                # if taxon contains '__' it is a morphotype and taxon is not equal to the one in the prev row
                if len(taxon.split('__'))>1:
                    print('morphology %s' %taxon)
                    na_values.append(taxon)

        #reload df with updated na_values and drop them
        print('reloading df')
        df = pd.read_csv(csv_path, na_values=na_values)
        print('dropping na')
        df = df.dropna()
        print('df size after clean %s' % (len(df)))

        #split lineage at '/', sort out all 'non-living', add rank
        #split
        df['lineage'] = df['lineage'].apply(split_lineage)

        # maximum linaeage, -3 because -'', -'#' and -'living'
        max_lineage = max(df['lineage'].apply(len))-3
        print('maximum lineage found: %s'%max_lineage)

        taxonomic_groups = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

        #sort out 'non-livning' by adding easily accessible column 'living'
        #create array
        living = [True if x[2]=='living' else False for x in df['lineage']]
        #and add as Series to df['living']
        df['living'] = pd.Series(living)
        #df of all living organisms
        living_df = df.loc[df['living']==True]

        # df['rank'] = np.nan
        #feed each taxon into ncbi db to get rank
        #if previous row has equal taxon, just copy the rank
        #first write all ranks into a list, attach this list as column in the end (speed up)
        prev_taxon = ''
        prev_rank = ''
        ranks = []
        for index, row in df.iterrows():
            print('updating rank in row: %s'%index)
            taxon = row['taxon']
            if prev_taxon == taxon:
                #copy rank
                # df['rank'][index] = prev_rank
                ranks.append(prev_rank)
            else:
                # look up taxon rank at ncbi and store it
                name2taxid = ncbi.get_name_translator([taxon])
                taxid = name2taxid[taxon]
                taxid2rank = ncbi.get_rank(taxid)
                # df['rank'][index] = taxid2rank[taxid[0]]
                rank = taxid2rank[taxid[0]]
                #replace curiosities like infrorder, subclass...
                if rank == 'subclass':
                    rank = 'class'
                if rank == 'infraorder':
                    rank = 'order'
                if rank == 'suborder':
                    rank = 'order'
                ranks.append(rank)
                prev_rank = rank
            prev_taxon = taxon
        print('adding ranks to df')
        df['rank'] = pd.Series(ranks)
        df = df.dropna()
        print('saving df')
        df.to_csv(file)
    else:
        df = pd.read_csv(file, na_values=missing_values_zoo)
        #load

    return df

def split_lineage(str):
    return str.split('/')

def plot_taxon_dist(x, y, rank, cc, cmap, colors, taxonomic_groups, taxonomic_groups_to_color, title):
    fig, ax = plt.subplots(figsize=(15, 6))
    taxon_plot = ax.bar(x, y, color=colors)

    # mapping for color bar
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(cc)))
    sm.set_array([])
    # color bar
    cbar = plt.colorbar(sm)
    cbar.set_label('Taxonomic Rank', rotation=270, labelpad=25)
    cbar.set_ticks([taxonomic_groups_to_color[x] - 0.0625 for x in taxonomic_groups])
    cbar.set_ticklabels(taxonomic_groups)

    # y ticks
    # for i, val in enumerate(y_fun):
    #     ax.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

    # x ticks
    plt.gca().set_xticklabels(x, rotation=60, horizontalalignment='right')
    # title
    plt.title(title, fontsize=22)
    # labels
    plt.ylabel("#")
    # plt.xlabel('Taxon')

    # make room for x labels
    plt.tight_layout()

def plot_class_dist(x, y, colors, title, axis=[]):
    if axis == []:
        fig, ax = plt.subplots(figsize=(15, 6))
        class_plot = ax.bar(x, y, color=colors)
    else:
        class_plot = axis.bar(x, y, color=colors)


    # y ticks
    # for i, val in enumerate(y_fun):
    #     ax.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

    # x ticks
    plt.gca().set_xticklabels(x, rotation=60, horizontalalignment='right')
    # title
    plt.title(title, fontsize=22)
    # labels
    plt.ylabel("#")
    # plt.xlabel('Taxon')

    # make room for x labels
    # plt.tight_layout()


def plot_fun(df_fun):
    print('plot fun')
    ##plot fun: taxonomic classes vs count with taxonomic rank colorcoded
    # classes
    x_fun = df_fun['taxon'].value_counts().index.tolist()
    # count
    y_fun = df_fun['taxon'].value_counts()
    # color coding taxonomic ranks
    taxonomic_groups = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomic_groups_to_color = {'kingdom': 0.875, 'phylum': 0.75, 'class': 0.625, 'order': 0.5, 'family': 0.375,
                                 'genus': 0.25, 'species': 0.125}
    #get maximum classification class
    rank = [df_fun.loc[df_fun['taxon'] == x]['rank'].tolist()[0] for x in x_fun]
    #color code maximum classification class
    cc = [taxonomic_groups_to_color[x] for x in rank]
    #receive color map
    cmap = plt.cm.get_cmap('Dark2', 7)
    #encode cc in cmap
    colors = cmap(cc)

    title = 'Taxonomic Distribution - Fungi Dataset'
    plot_taxon_dist(x=x_fun,
                    y=y_fun,
                    rank=rank,
                    cc=cc,
                    cmap=cmap,
                    colors=colors,
                    taxonomic_groups=taxonomic_groups,
                    taxonomic_groups_to_color=taxonomic_groups_to_color,
                    title=title)
    plt.savefig('Taxonomic_Distribution_-_Fungi_Dataset.png', bbox_inches='tight')
    ##plot fun:  classes vs count with taxonomic rank colorcoded
    #value count of rank
    raw_class_count = df_fun['rank'].value_counts()
    #transform to value count of rank according to taxonomic_groups ordering with 0 if no entry
    class_count = [raw_class_count[c] if c in raw_class_count else 0 for c in taxonomic_groups]
    #color encode
    cc = [taxonomic_groups_to_color[x] for x in taxonomic_groups]
    cmap = plt.cm.get_cmap('Dark2', 7)
    colors = cmap(cc)

    title = 'Class Distribution - Fungi Dataset'
    plot_class_dist(x=taxonomic_groups,
                    y=class_count,
                    colors=colors,
                    title=title)
    plt.tight_layout()
    plt.savefig('Class_Distribution_-_Fungi_Dataset.png', bbox_inches='tight')

    ##plot fun:  single class distribution with taxonomic rank colorcoded
    fig = plt.figure(figsize=(15, 10))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = 1
    for i, taxa in enumerate(taxonomic_groups):
        if taxa == 'kingdom': continue
        # title = '%s Distribution - Fungi Dataset'%taxa
        title=""
        #get sub dataframe where rank = taxa
        sub_df = df_fun.loc[df_fun['rank']==taxa]
        #set color accordingly
        color = colors[i]
        #get counts and plot
        if len(sub_df)>0:
            x = sub_df['taxon'].value_counts().index.tolist()
            # count
            y = sub_df['taxon'].value_counts()
            ax = fig.add_subplot(2, 3, subplot)
            plot_class_dist(x=x,
                            y=y,
                            colors=color,
                            title=title,
                            axis=ax)
            subplot+=1
            # ax.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')
    plt.tight_layout()
    plt.savefig('Within_Class_Distribution_-_Fungi_Dataset.png', bbox_inches='tight')


def plot_zoo(df_zoo):
    print('plot zoo')
    ##plot fun: taxonomic classes vs count with taxonomic rank colorcoded
    # classes
    x_zoo = df_zoo['taxon'].value_counts().index.tolist()
    # count
    y_zoo = df_zoo['taxon'].value_counts()

    # color coding taxonomic ranks
    taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomic_groups_to_color = { 'phylum': 0.857, 'class': 0.714, 'order': 0.571, 'family': 0.429,
                                 'genus': 0.286, 'species': 0.143}
    #get maximum classification class
    rank = [df_zoo.loc[df_zoo['taxon'] == x]['rank'].tolist()[0] for x in x_zoo]
    #color code maximum classification class
    cc = [taxonomic_groups_to_color[x] for x in rank]
    #receive color map
    cmap = plt.cm.get_cmap('Dark2', 6)
    #encode cc in cmap
    colors = cmap(cc)

    title = 'Taxonomic Distribution - ZooScan Dataset'
    plot_taxon_dist(x=x_zoo,
                    y=y_zoo,
                    rank=rank,
                    cc=cc,
                    cmap=cmap,
                    colors=colors,
                    taxonomic_groups=taxonomic_groups,
                    taxonomic_groups_to_color=taxonomic_groups_to_color,
                    title=title)
    plt.savefig('Taxonomic_Distribution_-_ZooScan_Dataset.png', bbox_inches='tight')
    ##plot zoo:  classes vs count with taxonomic rank colorcoded
    #value count of rank
    raw_class_count = df_zoo['rank'].value_counts()
    #transform to value count of rank according to taxonomic_groups ordering with 0 if no entry
    class_count = [raw_class_count[c] if c in raw_class_count else 0 for c in taxonomic_groups]
    #color encode
    cc = [taxonomic_groups_to_color[x] for x in taxonomic_groups]
    cmap = plt.cm.get_cmap('Dark2', 6)
    colors = cmap(cc)

    title = 'Class Distribution - ZooScan Dataset'
    plot_class_dist(x=taxonomic_groups,
                    y=class_count,
                    colors=colors,
                    title=title)
    plt.tight_layout()
    plt.savefig('Class_Distribution_-_ZooScan_Dataset.png', bbox_inches='tight')
    ##plot fun:  single class distribution with taxonomic rank colorcoded
    fig = plt.figure(figsize=(15, 10))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = 1
    for i, taxa in enumerate(taxonomic_groups):
        # title = '%s Distribution - Fungi Dataset'%taxa
        title=""
        #get sub dataframe where rank = taxa
        sub_df = df_zoo.loc[df_zoo['rank']==taxa]
        #set color accordingly
        color = colors[i]
        #get counts and plot
        if len(sub_df)>0:
            x = sub_df['taxon'].value_counts().index.tolist()
            # count
            y = sub_df['taxon'].value_counts()
            ax = fig.add_subplot(2, 3, subplot)
            plot_class_dist(x=x,
                            y=y,
                            colors=color,
                            title=title,
                            axis=ax)
            subplot+=1
            # ax.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')
    plt.tight_layout()
    plt.savefig('Within_Class_Distribution_-_ZooScan_Dataset.png', bbox_inches='tight')



# pathes to datasets that need to be analyzed and taxonomic classification file
path_fun = '/home/stillsen/Documents/Data/Image_classification_soil_fungi'
tax_file_fun = 'im_merged.csv'
path_zoo = '/home/stillsen/Documents/Data/ZooNet/ZooScanSet'
tax_file_zoo = 'taxa.csv'

#missing value definition
missing_values_fun = ['', 'unknown', 'unclassified']
missing_values_zoo = ['',
                      'artefact',
                      'bubble',
                      'detritus',
                      'seaweed',
                      'head',
                      'megalopa',
                      'Rhincalanidae',
                      'cirrus',
                      'metanauplii',
                      'cyphonaute',
                      'scale',
                      'Pyrosomatida',
                      'ephyra']

# getting Dataframe for Fungi and Zoo
csv_path = os.path.join(path_fun, tax_file_fun)
df_fun = pd.read_csv(csv_path, na_values=missing_values_fun)
# #add column taxon and rank, s.a.
df_fun = prepare_fun_df(df_fun)

csv_path = os.path.join(path_zoo, tax_file_zoo)
df_zoo = prepare_zoo_df(csv_path, na_values=missing_values_zoo)


## figures
plot_fun(df_fun)
plot_zoo(df_zoo)
plt.show()