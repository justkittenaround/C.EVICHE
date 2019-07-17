import os
from glob import glob
import csv
import numpy as np

os.chdir('/home/whale/Desktop/Rachel/CeVICHE/Test_Results')
fnames = glob('*/**.csv')
master = []
for name in fnames:
  with open(name, 'r') as f:
    reaer=csv.reader(f)
    data = list(reaer)
    master.append(data)
with open('master.csv', mode='w') as csv_file:
  writer = csv.writer(csv_file, delimiter=',')
  for info in master:
    writer.writerow(info)
