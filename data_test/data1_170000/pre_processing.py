# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:30:18 2019

@author: SJTUwwz
"""

import torch
from torch.utils.data import Dataset, DataLoader

class origin_dataset(Dataset):
    def __init__(self, file_name, begin, end):
        self.sentences = []
        self.labels = []
        self.vocabulary = []
        word_list = []
        f = open(file_name, 'r', encoding='utf-8')
        lines = f.readlines()
        words_list = []
        for i in range(begin, end):
            line = lines[i]
            line = line.rstrip()
            split_line = line.split(" : ")
            self.sentences.append(split_line[0])
            self.labels.append(split_line[1])
            words = split_line[0].split(" ")
            for word in words:
                words_list.append(word)
        self.vocabulary = set(words_list)
        
    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)
    
if __name__ == "__main__":
    test_dataset = origin_dataset("dataset.txt", 0, 10000)
    print(len(test_dataset.vocabulary))
    train_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    for se, la in train_loader:
        print(se,la)