import numpy as np
from torch.utils.data import Dataset
import os
import re

class FunctionSet(Dataset):
    def __init__(self, all_file_path, a3m_dir, repr_path=None, label_path=None):
        if repr_path is not None and label_path is not None:
            self.repr = np.load(repr_path)
            self.label = np.load(label_path)
            return
        if all_file_path == "data/Metal Ion Binding/test.txt":
            self.setType = "test"
            if a3m_dir == None:
                repr_list = []
                label_list = []
                for path, dir_list, file_list in os.walk("data/Metal Ion Binding/"):
                    for file_name in file_list:
                        if "test-reprs-" in file_name:
                            repr_list.append(np.load("data/Metal Ion Binding/"+file_name))
                        elif "test-labels-" in file_name:
                            label_list.append(np.load("data/Metal Ion Binding/"+file_name))
                self.repr = np.append(repr_list[0],repr_list[1:],axis=0)
                self.label = np.append(label_list[0],label_list[1:],axis=0)
                print(self.repr.shape)
                print(self.label.shape)
            else:
                self.a3m_dir = a3m_dir
                sequences, labels = self.read_data(all_file_path)
                self.repr, self.label = self.make_repr(sequences, labels)
        else:
            self.setType = "train"
            if a3m_dir == None:
                repr_list = []
                label_list = []
                for path, dir_list, file_list in os.walk("data/Metal Ion Binding/"):
                    for file_name in file_list:
                        if "reprs-" in file_name and "test" not in file_name:
                            repr_list.append(np.load("data/Metal Ion Binding/"+file_name))
                        elif "labels-" in file_name and "test" not in file_name:
                            label_list.append(np.load("data/Metal Ion Binding/"+file_name))
                self.repr = np.append(repr_list[0],repr_list[1:],axis=0)
                self.label = np.append(label_list[0],label_list[1:],axis=0)
                print(self.repr.shape)
                print(self.label.shape)
            else:
                self.a3m_dir = a3m_dir
                sequences, labels = self.read_data(all_file_path)
                self.repr, self.label = self.make_repr(sequences, labels)
        return
    
    def __getitem__(self, i):
        index = i % self.__len__()
        return self.repr[index], self.label[index]
    
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
                if os.path.splitext(file_name)[-1] != ".a3m":
                    continue
                file = open(os.path.join(self.a3m_dir, file_name))
                tags, seqs = self.parse_first_fasta(file.read())
                if seqs in sequence or sequence in seqs:
                    name = tags
                    if self.setType == "train":
                        repr_path = "data/Metal Ion Binding/repr/>" + name + "_msa_repr.npz.npy"
                    else:
                        repr_path = "data/Metal Ion Binding/test-repr/>" + name + "_msa_repr.npz.npy"
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
                repr = self.align_repr(repr)
                reprs.append(repr)
            index += 1
            if index % 100 == 0:
                print(f"Top {index} representations computed")
            if index % 1000 == 0 or index == len(sequences):
                part = index // 1000 - 1
                if index == len(sequences):
                    part += 1
                if self.setType == "train":
                    np.save(f"data/Metal Ion Binding/reprs-{part}.npz", np.array(reprs[part*1000:index]))
                    np.save(f"data/Metal Ion Binding/labels-{part}.npz", np.array(labels[part*1000:index]))
                else:
                    np.save(f"data/Metal Ion Binding/test-reprs-{part}.npz", np.array(reprs[part*1000:index]))
                    np.save(f"data/Metal Ion Binding/test-labels-{part}.npz", np.array(labels[part*1000:index]))
        return reprs, labels
    
    def align_repr(self, repr):
        residue_num = repr.shape[1]
        if residue_num > 256:
            new_repr=repr[:, 0:256, :]
        elif residue_num == 256:
            new_repr=repr
        else:
            new_repr=np.pad(repr,((0, 0),(0, 256-residue_num),(0, 0)))
        return new_repr