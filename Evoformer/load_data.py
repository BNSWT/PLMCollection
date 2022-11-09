import os
import numpy as np
from datetime import datetime
import re

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

def parse_fasta(data, name):
    data = re.sub('>$', '', data, flags=re.M)
    if '>' in data:
        lines = [
            l.replace('\n', '')
            for prot in data.split('>') for l in prot.strip().split('\n', 1)
        ][1:]
        tags, seqs = lines[::2], lines[1::2]
        tags = [t.split()[0] for t in tags]
        tags = tags[0]
        seqs = seqs[0]
    else:
        seqs = [data.replace('\n', '')]
        tags = [name.split('.')[0]]

    return tags, seqs

def make_fasta_from_a3m(a3m_dir):
    g = os.walk(a3m_dir)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            file = open(os.path.join(a3m_dir, file_name))
            tags, seqs = parse_fasta(file.read(), 'temp')

make_fasta_from_a3m("output/alignments")          