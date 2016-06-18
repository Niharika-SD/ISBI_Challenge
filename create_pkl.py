from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd

# from dir_to_dataset import dir_to_dataset
def dir_to_dataset(glob_files, loc_train_labels=""):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
        im = Image.open(file_name).convert('LA') #tograyscale
        img = im.resize((256,256), Image.ANTIALIAS)
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 10 == 0:
            print("\t %s files processed"%file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels)
        a = np.array(df)
        df = a.flatten()
        return np.array(dataset), df
    else:
        return np.array(dataset)


train_set_x, train_set_y = dir_to_dataset("train_data/*.jpg","trainLabels.csv")
val_set_x, val_set_y = dir_to_dataset("val_data/*.jpg","valLabels.csv")
test_set_x, test_set_y = dir_to_dataset("test_data/*.jpg","testLabels.csv")
# Data and labels are read 

print(val_set_y)
print(test_set_y.shape)
train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, val_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('file.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()





