import os
import pandas as pd

path = '/home/stillsen/Documents/Data/rec'
# missing_values = ['', 'unknown', 'unclassified']
missing_values = ['', 'unknown', 'unclassified', 'unidentified']
taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
# augment = 'transform'
oversample_technique = 'transformOversampled' # 'naiveOversampled', 'smote'
dataset = 'fun'
net_name = 'densenet169'

csv_path = os.path.join(path, 'im.merged.v10032020_unique_id_set.csv')
df = pd.read_csv(csv_path, na_values=missing_values)

print('NaNs in the label data set')
# print(df.isnull().sum())
print(df.iloc[:, 16:22].isnull().sum())

for i, row in df.iterrows():
    # if row has mv
    if row.iloc[16:22].isnull().any():
        if not pd.isnull(row.iloc[16:22][0]):
            lower_taxon = row.iloc[16:22][0]
        else:
            lower_taxon = 'fungi'
        for j, item in enumerate(row.iloc[16:22]):
            if pd.isnull(item):
                imp_taxon = lower_taxon + '_' + taxonomic_groups[j]
                df.iloc[i, j+16] = imp_taxon
                print('(%i,%i,)-> %s'%(i,j+16,df.iloc[i, j+16]))

            else:
                lower_taxon = item
print(df.iloc[:, 16:22].isnull().sum())
df.to_csv(os.path.join(path, 'im.merged.v10032020_unique_id_set__mv_imuted.csv'))