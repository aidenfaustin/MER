import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
from torch.utils.data import DataLoader

#mean std
import os
import re
import pickle, pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import  Dataset, DataLoader
from collections import defaultdict
from torch.utils.data import  Dataset, DataLoader

#dataset
class MELDRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        '''
        label index mapping =
        '''
        # robert features
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open('./meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')

        # comet features
        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
            = pickle.load(open('./meld/meld_features_comet.pkl', 'rb'), encoding='latin1')

        # audio_features
        self.audio_feature = defaultdict(list)

        # csv_path_train = '../../../Data/Meld/meld_opensmile/6552/train'
        csv_path_train = '/import/c4dm-datasets/jl007/Data/Meld/Wav/wav/fairseq/examples/data2vec/models/train'
        csv_files_train = os.listdir(csv_path_train)
        print('trainset',len(csv_files_train))
        csv_files_train.sort(key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1][3:-4])))

        for csv_file in csv_files_train:
            # with open(csv_path_train + '/' + csv_file) as f:
            #     last_line = f.readlines()[-1]  # ARFF格式csv文件最后一行包含特征数据
            # feature = last_line.split(",")
            # feature = np.array(feature[1:-1],
            #                    dtype="float64")
            # feature = (feature - mean)/(std+1e-5)
            features = np.load(csv_path_train + '/' + csv_file)
            features = features.squeeze(0)
            features = np.mean(features, axis = 0)
            #print('features',features.shape)
            feature = features.tolist()  
            # feature = np.array(feature[1:-1],
            #                    dtype="float64").tolist()
            k, nums = re.findall(r"\d+\d*", csv_file)
            # 第2到倒数第二个为特征数据，共384维特征
            self.audio_feature[int(k)].append(feature)

        # csv_path_dev = '../../../Data/Meld/meld_opensmile/6552/dev'
        csv_path_dev = '/import/c4dm-datasets/jl007/Data/Meld/Wav/wav/fairseq/examples/data2vec/models/dev'
        csv_files_dev = os.listdir(csv_path_dev)
        print('devset',len(csv_files_dev))
        csv_files_dev.sort(key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1][3:-4])))

        for csv_file in csv_files_dev:
            # with open(csv_path_dev + '/' + csv_file) as f:
            #     last_line = f.readlines()[-1]  # ARFF格式csv文件最后一行包含特征数据
            # feature = last_line.split(",")
            # feature = np.array(feature[1:-1],
            #                    dtype="float64")
            # feature = (feature - mean)/(std+1e-5)
            features = np.load(csv_path_dev + '/' + csv_file)
            features = features.squeeze(0)
            features = np.mean(features, axis = 0)
            #print('features',features.shape)
            feature = features.tolist()  
            # feature = np.array(feature[1:-1],
            #                    dtype="float64").tolist()
            k, nums = re.findall(r"\d+\d*", csv_file)

            # 第2到倒数第二个为特征数据，共384维特征
            self.audio_feature[int(k)+1039].append(feature)


        # csv_path_test = '../../../Data/Meld/meld_opensmile/6552/test'
        csv_path_test = '/import/c4dm-datasets/jl007/Data/Meld/Wav/wav/fairseq/examples/data2vec/models/test'
        csv_files_test = os.listdir(csv_path_test)
        print('testset',len(csv_files_test))
        csv_files_test.sort(key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1][3:-4])))

        for csv_file in csv_files_test:
            # with open(csv_path_test + '/' + csv_file) as f:
            #     last_line = f.readlines()[-1]  # ARFF格式csv文件最后一行包含特征数据
            # feature = last_line.split(",")
            # feature = np.array(feature[1:-1],
            #                    dtype="float64")
            # feature = (feature - mean)/(std+1e-5)
            features = np.load(csv_path_test + '/' + csv_file)
            features = features.squeeze(0)
            features = np.mean(features, axis = 0)
            #print('features',features.shape)
            feature = features.tolist()  
            # feature = np.array(feature[1:-1],
            #                    dtype="float64").tolist()
            k, nums = re.findall(r"\d+\d*", csv_file)
            # 第2到倒数第二个为特征数据，共384维特征
            self.audio_feature[int(k)+1153].append(feature)

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        # print(vid)
        # data[:-1]
        '''
        r1, r2, r3, r4, audio_feature\
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = data[:-1]

        '''

        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor(self.audio_feature[vid]), \
               torch.FloatTensor(self.xIntent[vid]), \
               torch.FloatTensor(self.xAttr[vid]), \
               torch.FloatTensor(self.xNeed[vid]), \
               torch.FloatTensor(self.xWant[vid]), \
               torch.FloatTensor(self.xEffect[vid]), \
               torch.FloatTensor(self.xReact[vid]), \
               torch.FloatTensor(self.oWant[vid]), \
               torch.FloatTensor(self.oEffect[vid]), \
               torch.FloatTensor(self.oReact[vid]), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 15 else pad_sequence(dat[i], True) if i < 17 else dat[i].tolist() for i in
                dat]

if __name__ == '__main__':

    trainset = MELDRobertaCometDataset('train', classify='emotion')
    train_loader = DataLoader(trainset,
                              batch_size=8,
                              collate_fn=trainset.collate_fn,
                              num_workers=0,
                              pin_memory=False)
    
    print(type(train_loader))

    # for data in train_loader:

    #     print(data[2].size())
    #     print(data[4].size())
    #     print(data[5].size())
    #     break

