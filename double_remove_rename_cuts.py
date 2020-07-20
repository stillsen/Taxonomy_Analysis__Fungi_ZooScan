import os, time
import pandas as pd
import shutil

path = "/home/stillsen/Documents/Data/Fungi_IC__new_set/cuts"
missing_values = ['', 'unknown', 'unclassified', 'unidentified']

# csv_path = os.path.join(path, 'im.merged.v10032020_back_scan_removed.csv')
# df = pd.read_csv(csv_path, na_values=missing_values)

# removed_file = 'EKA_4.12.17_E__4.12.17_E__cut__6.png'
idx = removed_file[-5]
print(idx)
max_idx = 13

#removing file
print('removing %s' %(os.path.join(path,removed_file)))
os.remove(os.path.join(path,removed_file))
# reindex the rest
for i in range(int(idx)+1,max_idx):
    filename = removed_file[:-5]+str(i)+removed_file[-4:]
    reindexed_filename = removed_file[:-5]+str(i-1)+removed_file[-4:]
    print('%s -> %s' %(filename, reindexed_filename))
    shutil.move(os.path.join(path,filename),os.path.join(path,reindexed_filename))
    # if file[-3:] == 'tif':
    #     mdate = time.gmtime(os.path.getmtime(os.path.join(path, file)))
    #     if delimiter_mdate < mdate: # file is younger
    #         if file[-7:-4] == '002': # scan from front -> keep and work on
    #             new_name = file[:6]+file[8:12]+file[-4:]
    #             os.rename(os.path.join(path,file),os.path.join(path,new_name))
    #             # apperently renaming a file does not change mdate, so I do not need to set it
    #         else: # scan from back -> dump
    #             shutil.move(os.path.join(path,file),os.path.join(excluded_path,file))
    #
