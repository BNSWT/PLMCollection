import os
import numpy as np
from datetime import datetime

filepath = "data/Metal Ion Binding/train.txt"

def read_data(filepath):
    sequences = []
    labels = []
    for line in open(filepath).readlines():
        sequences.append(line.split('\t')[0])
        labels.append(line.split('\t')[1].strip('\n'))
    return sequences, labels

def make_fasta(sequences, dir):
    print(len(sequences))
    iter = 0
    for sequence in sequences:
        tag = str(int(round(datetime.now().timestamp()))+iter)
        iter += 1
        open(dir+tag+".fasta", "w").write(sequence)

def make_x(path):
    x = []
    g = os.walk(path)  
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            msa_repr=np.load(os.path.join(path, file_name))
            x.append(msa_repr)
    return np.array(x)
            
def make_y(labels):
    return np.array(labels)