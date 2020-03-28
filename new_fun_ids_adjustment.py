import os, time
import pandas as pd
import shutil

path = "/home/stillsen/Documents/Data/Fungi_IC__new_set"
missing_values = ['', 'unknown', 'unclassified', 'unidentified']

csv_path = os.path.join(path, 'im.merged.v10032020_back_scan_removed.csv')
df = pd.read_csv(csv_path, na_values=missing_values)

# preparing the mdate_skey unique identifiers for the new dataset to fit the old data set
mdate_skey_col = []
pos_scan_col = []
for index, row in df.iterrows():
    pos_scan_col.append(row['Pos_scan'])
    # correct mdate_skey, only copy to new col
    if len(row['Scan_file'].split('_')) > 1:
        mdate_skey_col.append(row['Scan_file'])
    else:
        # there is mdate in 'Scan_date' and skey + and identifier for front (002) or back (001)
        if row['Scan_file'][-1] == str(2): # only forderseite
            new_mdate = row['Scan_date'][:-4]+row['Scan_date'][-2:]
            new_mdate_skey = new_mdate+'_'+row['Scan_file'][0]
            mdate_skey_col.append(new_mdate_skey)

df['Scan.date'] = mdate_skey_col
df['Pos.scan'] = pos_scan_col
csv_path = os.path.join(path,'im.merged.v10032020_unique_id_set.csv' )
df.to_csv(csv_path)

# changing the new file set to meet old standards,
# only tif
# throw out back side (001)
# every file younger than delimiter file

delimeter_file = 'MOM_EX_A (4).tif'
delimiter_mdate = time.gmtime(os.path.getmtime(os.path.join(path, delimeter_file)))
excluded_path = os.path.join(path,'excluded')
filenames = os.listdir(path)
for file in filenames:
    if file[-3:] == 'tif':
        mdate = time.gmtime(os.path.getmtime(os.path.join(path, file)))
        if delimiter_mdate < mdate: # file is younger
            if file[-7:-4] == '002': # scan from front -> keep and work on
                new_name = file[:6]+file[8:12]+file[-4:]
                os.rename(os.path.join(path,file),os.path.join(path,new_name))
                # apperently renaming a file does not change mdate, so I do not need to set it
            else: # scan from back -> dump
                shutil.move(os.path.join(path,file),os.path.join(excluded_path,file))

