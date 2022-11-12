import numpy as np
from torch.utils.data import Dataset
import os
import re

class FunctionSet(Dataset):
    def __init__(self, all_file_path, a3m_dir):
        self.a3m_dir = a3m_dir
        sequences, labels = self.read_data(all_file_path)
        self.repr, self.labels = self.make_repr(sequences, labels)
        return
    
    def __getitem__(self, i):
        index = i % self.__len__()
        return self.repr[index], self.labels[index]
    
    def __len__(self):
        return len(self.repr)
       
    def read_data(self, filepath):
        sequences = []
        labels = []
        for line in open(filepath).readlines():
            sequences.append(line.split('\t')[0])
            labels.append(int(line.split('\t')[1].strip('\n')))
        return sequences, labels
        
    def parse_first_fasta(self, data):
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
        return tags, seqs
    
    def find_repr(self, sequence):
        g = os.walk(self.a3m_dir)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                file = open(os.path.join(self.a3m_dir, file_name))
                tags, seqs = self.parse_first_fasta(file.read())
                if seqs in sequence or sequence in seqs:
                    name = tags
                    repr_path = "data/Metal Ion Binding/repr/>" + name + "_msa_repr.npz.npy"
                    if os.path.exists(repr_path):
                        return np.load(repr_path)
                    else:
                        print(f"repr not found. tag:{tags}, sequence:{seqs}")
        print("Not found")
        return None
    
    def make_repr(self, sequences, labels):
        reprs = []
        index = 0
        for sequence in sequences:
            repr = self.find_repr(sequence)
            if repr is None:
                del labels[index]
            else:
                reprs.append(repr)
            index += 1
            # if index == 10:
            #     break
            
        reprs = self.align_reprs(reprs)
        np.save("data/Metal Ion Binding/reprs.npz", np.array(reprs))
        np.save("data/Metal Ion Binding/labels.npz", np.array(labels))
        return reprs, labels
    
    def align_reprs(self, reprs):
        new_reprs = []
        for repr in reprs:
            residue_num = repr.shape[1]
            if residue_num > 256:
                new_reprs.append(repr[:, 0:256, :])
            elif residue_num == 256:
                new_reprs.append(repr)
            else:
                new_reprs.append(np.pad(repr,((0, 0),(0, 256-residue_num),(0, 0))))
        return new_reprs