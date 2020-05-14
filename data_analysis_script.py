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
from collections import Counter

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
                # df['taxon'][index] = df[higher_taxa][index]
                df[index, 'taxon'] = df[higher_taxa][index]
                rank = higher_taxa
            #set the maximum classification class
            # df['rank'][index] = rank
            df.ix[index, 'rank'] = rank
            if not pd.isnull(df['taxon'][index]):
                break
    return df

def split_lineage(str):
    return str.split('/')

def plot_taxon_dist(x, y, cc, cmap, colors, taxonomic_groups, taxonomic_groups_to_color, title, offset = 0.0):
    fig, ax = plt.subplots(figsize=(15, 6))
    taxon_plot = ax.bar(x, y, color=colors)

    # mapping for color bar
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(cc)))
    sm.set_array([])
    # color bar
    cbar = plt.colorbar(sm)
    cbar.set_label('Taxonomic Rank', rotation=270, labelpad=25)
    # cbar.set_ticks([taxonomic_groups_to_color[x] - 0.0625 for x in taxonomic_groups])
    cbar.set_ticks([taxonomic_groups_to_color[x] + offset for x in taxonomic_groups])
    cbar.set_ticklabels(taxonomic_groups)

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # y ticks
    # for i, val in enumerate(y_fun):
    #     ax.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

    # x ticks
    # plt.gca().set_xticklabels(x, rotation=60, horizontalalignment='right')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    # title
    # plt.title(title, fontsize=22)
    ax.text(.5, .9, title, horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    # labels
    plt.ylabel("#")
    # plt.xlabel('Taxon')

    # make room for x labels
    # plt.tight_layout()

def plot_class_dist(x, y, colors, title, label_x, ax=None):
    if ax == None:
        fig, ax = plt.subplots(figsize=(15, 6))
        class_plot = ax.bar(x, y, color=colors)
    else:
        class_plot = ax.bar(x, y, color=colors)


    # y ticks
    # for i, val in enumerate(y_fun):
    #     ax.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})
    if not label_x:
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
    else:
        # x ticks
        plt.gca().set_xticklabels(x, rotation=60, horizontalalignment='right')
        # title
        # plt.title(title, fontsize=22)

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(.9, .95, title, horizontalalignment='center', transform=ax.transAxes, fontsize=12)
    # labels
    plt.ylabel("#")
    # plt.xlabel('Taxon')

    # make room for x labels
    # plt.tight_layout()


def plot_fun(df_fun, figure_path):
    print('plot fun')
    ##plot fun: taxonomic classes vs count with taxonomic rank colorcoded
    # color coding taxonomic ranks
    taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomic_groups_to_color = { 'phylum': 0.857142857142857, 'class': 0.714285714285714, 'order': 0.571428571428571, 'family': 0.428571428571429,
                                 'genus':  0.285714285714286, 'species': 0.142857142857143}

    x_fun = []
    y_fun = []
    ranks = []
    for i, rank in enumerate(taxonomic_groups):
        # classes
        x_rank = df_fun[rank].value_counts().index.tolist()
        x_fun.extend(x_rank)
        # count
        y_rank = df_fun[rank].value_counts()
        y_fun.extend(y_rank)
        # rank vector
        n_rank = [rank]*len(y_rank)
        # print('extending by factor %i, and is of length %i' % (len(y_fun), len(n_rank)))
        ranks.extend(n_rank)
        # print('stop')


    #get maximum classification class
    # rank = [df_fun.loc[df_fun['taxon'] == x]['rank'].tolist()[0] for x in x_fun]
    #color code maximum classification class
    cc = [taxonomic_groups_to_color[x] for x in ranks]
    #receive color map
    cmap = plt.cm.get_cmap('Dark2', 6)
    #encode cc in cmap
    colors = cmap(cc)

    # 'Per-Level Taxonomic Distribution - Fungi Data set'
    title = ''
    plot_taxon_dist(x=x_fun,
                    y=y_fun,
                    cc=cc,
                    cmap=cmap,
                    colors=colors,
                    taxonomic_groups=taxonomic_groups,
                    taxonomic_groups_to_color=taxonomic_groups_to_color,
                    title=title,
                    offset=-(0.142857142857143/2))
    fig_file = os.path.join(figure_path, 'data__per-lvl_tdist-fds.png')
    plt.savefig(fig_file, bbox_inches='tight')

    ##plot fun:  classes vs count with taxonomic rank colorcoded
    #value count of rank
    x = Counter(ranks).keys()
    y = Counter(ranks).values()
    #color encode
    cc = [taxonomic_groups_to_color[xi] for xi in x]
    cmap = plt.cm.get_cmap('Dark2', 6)
    colors = cmap(cc)

    # 'Unique Class Count Per Taxonomic Rank - Fungi Data set'
    title = ''
    plot_class_dist(x=x,
                    y=y,
                    colors=colors,
                    title=title,
                    label_x = True)
    plt.tight_layout()
    fig_file = os.path.join(figure_path, 'data__ucc_-_fds.png')
    plt.savefig(fig_file, bbox_inches='tight')

    ##plot fun:  single class distribution with taxonomic rank colorcoded
    fig = plt.figure(figsize=(15, 10))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = 1
    for i, rank in enumerate(taxonomic_groups):
        # get counts per rank as dict
        d = Counter(df_fun[rank])
        # converting nan to "unclassified"
        if np.nan in d:
            d['unclassified'] = d.pop(np.nan)
        # title = '%s Distribution - Fungi Dataset'%taxa
        title = str(rank)
        # title = ''
        #set color accordingly
        color = colors[i]

        ax = fig.add_subplot(2, 3, subplot)
        plot_class_dist(x=d.keys(),
                        y=d.values(),
                        colors=color,
                        title=title,
                        label_x = False,
                        ax=ax)
        subplot+=1
            # ax.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')
    plt.tight_layout()
    fig_file = os.path.join(figure_path, 'data__rank_dist_-_fds.png')
    plt.savefig(fig_file, bbox_inches='tight')

def plot_fun_classificationSL(path,  figure_path, metric = 'pcc'):
    print('plot fun classification')

    # storage_folder = 'ParameterFiles_densenet169_e%i_lr0.001000_m0.800000'%epochs
    # storage_path = os.path.join(storage_path,storage_folder)
    # dataset = 'fun'
    # mode = 'per_lvl'

    subdirs = [x[0] for x in os.walk(path)][1:]

    # color coding taxonomic ranks
    taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomic_groups_to_color = { 'phylum': 0.857142857142857, 'class': 0.714285714285714, 'order': 0.571428571428571, 'family': 0.428571428571429, 'genus':  0.285714285714286, 'species': 0.142857142857143}

    cc = [taxonomic_groups_to_color[x] for x in taxonomic_groups]
    #receive color map
    cmap = plt.cm.get_cmap(name='Dark2', lut=6)
    # cmap = plt.cm.get_cmap(name='Paired', lut=12)
    #encode cc in cmap
    colors = cmap(cc)
    # colors = cmap.colors

    # all SL
    fig = plt.figure(figsize=(15, 10))
    subplot = 1
    # abc = iter(['a','b','c','d'])
    abc = iter(['a) original', 'b) simple oversampled', 'c) transform oversampled', 'd'])
    l = []
    firstrun = True
    for subdir in sorted(subdirs):
        print('working in %s'%subdir)
        if subdir.split('/')[-1] != 'figures' and 'ML' not in subdir.split('/')[-1] and 'oversamplebias' not in subdir.split('/')[-1]:
            ax = fig.add_subplot(2, 2, subplot)

            # title = next(abc)
            title = next(abc) #+') '+ subdir.split('/')[-1].split('_')[1] +' '+ subdir.split('/')[-1].split('_')[2]
            i = 0
            if 'tt-split' in subdir.split('/')[-1]:
                for file in [x[-1] for x in os.walk(subdir)][0]:
                    if metric == 'acc':
                        if 'acc' in file:
                            print('reading %s'%file)
                            df = pd.read_csv(os.path.join(subdir,file))
                            line_train, = plt.plot('epochs', 'acc_train', data=df, marker='', color=colors[i],
                                                   linewidth=1, label=file.split('_')[-2])
                            line_test, = plt.plot('epochs', 'acc_test', data=df, marker='', color=colors[i],
                                                  linewidth=1)
                        else: continue
                    else:
                        if 'acc' not in file:
                            print('reading %s'%file)
                            df = pd.read_csv(os.path.join(subdir,file))
                            line_train, = plt.plot('epochs', 'scores_train', data=df, marker='', color=colors[i],
                                                   linewidth=1, label=file.split('_')[-2])
                            line_test, = plt.plot('epochs', 'scores_test', data=df, marker='', color=colors[i],
                                                  linewidth=1)
                        else: continue


                    i+=1
                    if firstrun: l.append(line_train)
                firstrun = False
            elif 'xval' in subdir.split('/')[-1]:
                d = dict()
                for file in [x[-1] for x in os.walk(subdir)][0]:
                    if (metric == 'acc' and 'acc' in file):
                        print('reading %s' % file)
                        if file.split('_')[-4] not in d:
                            d[file.split('_')[-4]] = pd.read_csv(os.path.join(subdir, file))
                        else:
                            d[file.split('_')[-4]] = pd.concat(
                                (d[file.split('_')[-4]], pd.read_csv(os.path.join(subdir, file))))
                    if (metric == 'pcc' and 'acc' not in file):
                        print('reading %s'%file)
                        if file.split('_')[-2] not in d:
                            d[file.split('_')[-2]] = pd.read_csv(os.path.join(subdir,file))
                        else:
                            d[file.split('_')[-2]] = pd.concat((d[file.split('_')[-2]], pd.read_csv(os.path.join(subdir,file))))
                dodge = 0
                for key, value in d.items(): # loop over ranks
                    by_row_index = value.groupby(value.index)
                    d[key] = by_row_index.mean()
                    d[key]['epochs'] = d[key]['epochs']+dodge
                    dodge+=0.05
                    if metric == 'pcc':
                        d[key]['train_sem'] = by_row_index.sem()['scores_train']
                        d[key]['test_sem'] = by_row_index.sem()['scores_test']
                        line_train = plt.errorbar('epochs', 'scores_train', yerr='train_sem', data=d[key], marker='', color=colors[i], linewidth=1)
                        line_test = plt.errorbar('epochs', 'scores_test', yerr='test_sem', data=d[key], marker='', color=colors[i], linewidth=1)
                    elif metric == 'acc':
                        d[key]['train_sem'] = by_row_index.sem()['acc_train']
                        d[key]['test_sem'] = by_row_index.sem()['acc_test']
                        line_train = plt.errorbar('epochs', 'acc_train', yerr='train_sem', data=d[key], marker='',
                                                  color=colors[i], linewidth=1)
                        line_test = plt.errorbar('epochs', 'acc_test', yerr='test_sem', data=d[key], marker='',
                                                 color=colors[i], linewidth=1)
                    i+=1


            if metric == 'acc':
                ax.set_ylabel('ACC', fontsize=12)
                plt.ylim(0, 1)
                plt.yticks([0, 0.25, 0.5, 1])
                # fig.suptitle('Accuracy', x=0.18, y=0.97,fontsize= 20)
            else:
                ax.set_ylabel('PCC/MCC', fontsize=12)
                plt.ylim(-1, 1)
                plt.yticks([-1, -0.5, 0, 0.1, 0.25, 0.5, 1])
                # fig.suptitle('MCC/PCC', x=0.18, y=0.97,fontsize= 20)
            plt.xticks([0, 5, 10, 15, 20])
            ax.set_xlabel('epochs', fontsize=12)

            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                #bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                #labelbottom=False
                )

            # removing top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # title
            if subplot == 1:
                ax.text(.92, 0.05, title, horizontalalignment='center', transform=ax.transAxes, fontsize=12)
            else:
                ax.text(.84, 0.05, title, horizontalalignment='center', transform=ax.transAxes, fontsize=12)

            # adds major gridlines
            ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.6)

            #add a legend
            # ax.legend([line_train, line_test], ['train','test'])
            if subplot == 1:
                # ax = fig.add_subplot(2, 2, subplot)
                ax.legend(l,taxonomic_groups)
            subplot += 1

            plt.tight_layout()
            fig_file = os.path.join(figure_path, 'performancePlot_SL_'+metric+'.png')
            # fig_file = os.path.join(figure_path, 'stabilityPlot_SL_' + metric + '.png')
            plt.savefig(fig_file, bbox_inches='tight')

def plot_fun_classificationML(path, figure_path, metric='pcc'):
    print('plot fun classification')

    # storage_folder = 'ParameterFiles_densenet169_e%i_lr0.001000_m0.800000'%epochs
    # storage_path = os.path.join(storage_path,storage_folder)
    # dataset = 'fun'
    # mode = 'per_lvl'
    plot_length = 60

    subdirs = [x[0] for x in os.walk(path)][1:]

    # color coding taxonomic ranks
    taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    tcn = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    taxonomic_groups_to_color = {'phylum': 0.857142857142857, 'class': 0.714285714285714,
                                 'order': 0.571428571428571, 'family': 0.428571428571429,
                                 'genus': 0.285714285714286, 'species': 0.142857142857143}

    cc = [taxonomic_groups_to_color[x] for x in taxonomic_groups]
    # receive color map
    cmap = plt.cm.get_cmap(name='Dark2', lut=6)
    # cmap = plt.cm.get_cmap(name='Paired', lut=12)
    # encode cc in cmap
    colors = cmap(cc)
    # colors = cmap.colors

    # all SL
    fig = plt.figure(figsize=(15, 10))
    subplot = 1
    abc = iter(['a) original', 'b) simple oversampled', 'c) transform oversampled', 'd'])
    l = []
    firstrun = True
    for subdir in sorted(subdirs):
        print('working in %s' % subdir)
        if subdir.split('/')[-1] != 'figures' and 'SL' not in subdir.split('/')[-1] and 'oversamplebias' not in subdir.split('/')[-1]:
            ax = fig.add_subplot(2, 2, subplot)

            # title = next(abc)
            title = next(abc) #+ ') ' + subdir.split('/')[-1].split('_')[1] + ' ' + subdir.split('/')[-1].split('_')[2]
            i = 0
            if 'tt-split' in subdir.split('/')[-1]:
                for file in [x[-1] for x in os.walk(subdir)][0]:
                    if metric == 'acc':
                        if 'acc' in file:
                            print('reading %s' % file)
                            df = pd.read_csv(os.path.join(subdir, file))
                            for rank in range(6):
                                col_name = '%s_Train_ACC'%tcn[rank]
                                print(col_name)
                                line_train, = plt.plot(df['Epochs'].to_list()[:plot_length], df[col_name].to_list()[:plot_length], marker='', color=colors[rank],linewidth=1, label=tcn[rank])
                                l.append(line_train)
                                col_name = '%s_Test_ACC' % (tcn[rank])
                                line_test, = plt.plot(df['Epochs'].to_list()[:plot_length], df[col_name].to_list()[:plot_length], marker='', color=colors[rank], linewidth=1, label=tcn[rank])
                        else:
                            continue
                    else:
                        if 'acc' not in file:
                            print('reading %s' % file)
                            df = pd.read_csv(os.path.join(subdir, file))
                            for rank in range(6):
                                col_name = '%s_Train_PCC'%tcn[rank]
                                print(col_name)
                                line_train, = plt.plot(df['Epochs'].to_list()[:plot_length], df[col_name].to_list()[:plot_length], marker='', color=colors[rank],linewidth=1, label=tcn[rank])
                                l.append(line_train)
                                col_name = '%s_Test_PCC' % (tcn[rank])
                                line_test, = plt.plot(df['Epochs'].to_list()[:plot_length], df[col_name].to_list()[:plot_length], marker='', color=colors[rank], linewidth=1, label=tcn[rank])


                        else:
                            continue

                    i += 1
                    # if firstrun: l.append(line_train)
                firstrun = False
            elif 'xval' in subdir.split('/')[-1]:
                d = dict()
                for file in [x[-1] for x in os.walk(subdir)][0]:
                    if (metric == 'acc' and 'acc' in file):
                        print('reading %s' % file)
                        if file.split('_')[-4] not in d:
                            d[file.split('_')[-4]] = pd.read_csv(os.path.join(subdir, file))
                        else:
                            d[file.split('_')[-4]] = pd.concat(
                                (d[file.split('_')[-4]], pd.read_csv(os.path.join(subdir, file))))
                    if (metric == 'pcc' and 'acc' not in file):
                        print('reading %s' % file)
                        if file.split('_')[-2] not in d:
                            d[file.split('_')[-2]] = pd.read_csv(os.path.join(subdir, file))
                        else:
                            d[file.split('_')[-2]] = pd.concat(
                                (d[file.split('_')[-2]], pd.read_csv(os.path.join(subdir, file))))
                dodge = 0
                for key, value in d.items():  # loop over ranks
                    by_row_index = value.groupby(value.index)
                    d[key] = by_row_index.mean()
                    d[key]['Epochs'] = d[key]['Epochs'] + dodge
                    dodge += 0.05
                    if metric == 'pcc':
                        for rank in range(6):
                            col_name = '%s_Train_PCC' % tcn[rank]
                            sem_name = '%s_Train_SEM' % tcn[rank]
                            d[key][sem_name] = by_row_index.sem()[col_name]
                            print(col_name)
                            line_train = plt.errorbar(d[key]['Epochs'].to_list()[:plot_length], d[key][col_name].to_list()[:plot_length], yerr=d[key][sem_name].to_list()[:plot_length], marker='', color=colors[rank], linewidth=1)

                            col_name = '%s_Test_PCC' % (tcn[rank])
                            sem_name = '%s_Test_SEM' % tcn[rank]
                            d[key][sem_name] = by_row_index.sem()[col_name]
                            line_test = plt.errorbar(d[key]['Epochs'].to_list()[:plot_length], d[key][col_name].to_list()[:plot_length], yerr=d[key][sem_name].to_list()[:plot_length], marker='',color=colors[rank], linewidth=1)
                    elif metric == 'acc':
                        for rank in range(6):
                            col_name = '%s_Train_ACC' % tcn[rank]
                            sem_name = '%s_Train_SEM' % tcn[rank]
                            d[key][sem_name] = by_row_index.sem()[col_name]
                            print(col_name)
                            line_train = plt.errorbar(d[key]['Epochs'].to_list()[:plot_length], d[key][col_name].to_list()[:plot_length], yerr=d[key][sem_name].to_list()[:plot_length], marker='', color=colors[rank], linewidth=1)

                            col_name = '%s_Test_ACC' % (tcn[rank])
                            sem_name = '%s_Test_SEM' % tcn[rank]
                            d[key][sem_name] = by_row_index.sem()[col_name]
                            line_test = plt.errorbar(d[key]['Epochs'].to_list()[:plot_length], d[key][col_name].to_list()[:plot_length], yerr=d[key][sem_name].to_list()[:plot_length], marker='',color=colors[rank], linewidth=1)
                    i += 1

            if metric == 'acc':
                ax.set_ylabel('ACC', fontsize=12)
                plt.ylim(0, 1)
                plt.yticks([0, 0.25, 0.5, 1])
                # fig.suptitle('Accuracy', x=0.18, y=0.97,fontsize= 20)
            else:
                ax.set_ylabel('PCC/MCC', fontsize=12)
                plt.ylim(-1, 1)
                plt.yticks([-1, -0.5, 0, 0.1, 0.25, 0.5, 1])
                # fig.suptitle('MCC/PCC', x=0.18, y=0.97,fontsize= 20)
            plt.xticks([0, 5, 10, 15, 20])
            ax.set_xlabel('epochs', fontsize=12)

            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                # bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                # labelbottom=False
            )

            # removing top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # title
            if subplot == 1:
                ax.text(.92, 0.05, title, horizontalalignment='center', transform=ax.transAxes, fontsize=12)
            else:
                ax.text(.84, 0.05, title, horizontalalignment='center', transform=ax.transAxes, fontsize=12)

            # adds major gridlines
            ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.6)

            # add a legend
            # ax.legend([line_train, line_test], ['train','test'])
            if subplot == 1:
                taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
                ax.legend(l, taxonomic_groups)
            subplot += 1

            plt.tight_layout()
            # fig_file = os.path.join(figure_path, 'stabilityPlot_ML_' + metric + '.png')
            fig_file = os.path.join(figure_path, 'performancePlot_ML_' + metric + '.png')
            plt.savefig(fig_file, bbox_inches='tight')

def get_comparisonDF(path, metric):
    df_cmp = pd.DataFrame(columns=['Score','Metric','Dataset','Model','Rank'])

    subdirs = [x[0] for x in os.walk(path)][1:]
    for subdir in sorted(subdirs):
        if 'orig' in subdir.split('/')[-1]: dataset = 'orig'
        elif 'naiveOversampled' in subdir.split('/')[-1]: dataset = 'naiveOversampled'
        elif 'transformOversampled' in subdir.split('/')[-1]: dataset = 'transformOversampled'

        if 'SL' in subdir:
            model = 'SL'
        if 'HC' in subdir:
            model = 'HC'
        if 'SL' in subdir or 'HC' in subdir:
            for file in [x[-1] for x in os.walk(subdir)][0]:
                if metric == 'acc':
                    if 'acc' in file:
                        print('reading %s' % file)
                        df = pd.read_csv(os.path.join(subdir, file))
                    else:
                        continue
                else:
                    if 'acc' not in file: # PCC
                        print('reading %s' % file)
                        df = pd.read_csv(os.path.join(subdir, file))
                    else:
                        continue
                score = df['scores_test'].max()
                rank = file.split('_')[-2]
                df_cmp = df_cmp.append({'Score': score, 'Metric': metric, 'Dataset': dataset, 'Model': model, 'Rank': rank},ignore_index=True)

        elif 'ML' in subdir:
            model = 'ML'
            for file in [x[-1] for x in os.walk(subdir)][0]:
                if metric == 'acc':
                    if 'acc' in file:
                        print('reading %s' % file)
                        df = pd.read_csv(os.path.join(subdir, file))
                    else:
                        continue
                else:
                    if 'acc' not in file:  # PCC
                        print('reading %s' % file)
                        df = pd.read_csv(os.path.join(subdir, file))
                        score = df['Phylum_Test_PCC'].max()
                        df_cmp = df_cmp.append(
                            {'Score': score, 'Metric': metric, 'Dataset': dataset, 'Model': model, 'Rank': 'phylum'},
                            ignore_index=True)
                        score = df['Class_Test_PCC'].max()
                        df_cmp = df_cmp.append(
                            {'Score': score, 'Metric': metric, 'Dataset': dataset, 'Model': model, 'Rank': 'class'},
                            ignore_index=True)
                        score = df['Order_Test_PCC'].max()
                        df_cmp = df_cmp.append(
                            {'Score': score, 'Metric': metric, 'Dataset': dataset, 'Model': model, 'Rank': 'order'},
                            ignore_index=True)
                        score = df['Family_Test_PCC'].max()
                        df_cmp = df_cmp.append(
                            {'Score': score, 'Metric': metric, 'Dataset': dataset, 'Model': model, 'Rank': 'family'},
                            ignore_index=True)
                        score = df['Genus_Test_PCC'].max()
                        df_cmp = df_cmp.append(
                            {'Score': score, 'Metric': metric, 'Dataset': dataset, 'Model': model, 'Rank': 'genus'},
                            ignore_index=True)
                        score = df['Species_Test_PCC'].max()
                        df_cmp = df_cmp.append(
                            {'Score': score, 'Metric': metric, 'Dataset': dataset, 'Model': model, 'Rank': 'species'},
                            ignore_index=True)
                    else:
                        continue
    return df_cmp

def comparisonPlot(path, figure_path, metric='pcc'):
    # get data
    df = get_comparisonDF(path,metric)
    df.to_csv(os.path.join(path,'comparisonPlot.csv'))
    # GGPLOT2 HERE WE COME
    # subdirs = [x[0] for x in os.walk(path)][1:]



# pathes to datasets that need to be analyzed and taxonomic classification file
# path_fun = '/home/stillsen/Documents/Data/Image_classification_soil_fungi'
# path_fun = "/home/stillsen/Documents/Data/Fungi_IC__new_set"
# tax_file_fun = 'im.merged.v10032020_unique_id_set.csv'

# epochs = 20
# storage_path = '/media/stillsen/Elements SE/Data/'

# path='/home/stillsen/Documents/Data/Results/PerformancePlot_SL'
path='/home/stillsen/Documents/Data/Results/ComparisionPlot'
# path='/home/stillsen/Documents/Data/Results/PerformancePlot_ML'
# path='/home/stillsen/Documents/Data/Results/StabilityPlot_SL'
#missing value definition
missing_values_fun = ['', 'unknown', 'unclassified', 'unidentified']

# # getting Dataframe for Fungi and Zoo
# csv_path = os.path.join(path_fun, tax_file_fun)
# df_fun = pd.read_csv(csv_path, na_values=missing_values_fun)
# # #add column taxon and rank, s.a.
# df_fun = prepare_fun_df(df_fun)

# ## figures
#plot_fun(df_fun=df_fun,figure_path=figure_path)
# plot_fun_classificationSL(path=path,  figure_path=path, metric='pcc')
comparisonPlot(path=path, figure_path=path, metric='pcc')
# plot_fun_classificationML(path=path,  figure_path=path, metric='pcc')
plt.show()