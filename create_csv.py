from glob import glob
import os
import csv

my_folder = 'home/kliv/Documents/train_data/'
files_list = glob(os.path.join(my_folder, '*.jpg'))
my_class = []
for a_file in sorted(files_list):
	temp = a_file.split("_")
	if(temp[3] == 'CH'):
		my_class.append(0)
	elif(temp[3] == 'NR'): 
		my_class.append(1)
	elif(temp[3] == 'VA'):
		my_class.append(2)
	elif(temp[3] == 'VACH'):
		my_class.append(3)
#print(my_class[1])
with open('trainLabels.csv', 'wb') as myfile:
	writer = csv.writer(myfile, delimiter=',',  quoting=csv.QUOTE_MINIMAL)
	writer.writerow(my_class)

import numpy as np
np.savetxt("file_name.csv", my_class, delimiter=",", fmt='%d', header=" ")


# f = open('trainLabels.csv', 'wt')
# try:
#     writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
#     writer.writerow( () )
#     for i in my_class:
#         writer.writerow( i );
# finally:
#     f.close()

# print open(my_folder, 'rt').read()