from torch.utils.data import Dataset, DataLoader
import os
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import json
import torch
import numpy as np
import torch.nn.functional as F
import random

class MVPDataset(Dataset):
    def __init__(self, feature_path, anno_path, confidence_path, subset='train'):
        self.anno_path = anno_path
        self.feature_path = feature_path
        self.subset = subset
        self.confidence_path = confidence_path
        self.transform = self.get_transform()
        self.annotation = self.get_annotation(anno_path)
        self.feature_path_list, self.label_list, self.confidence_list, self.key_list = self.get_instance(self.annotation)

        self.choose_list =  self.feature_path_list.copy()
        self.voter_number =  1
     
    def get_annotation(self, root_path):
        
        with open(root_path, 'r') as f:
            anno = json.load(f)
        return anno
        # return image_path_list

    def get_transform(self):
        return transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
        ])

    def get_instance(self, anno):
        label_list = []
        feat_path_list = []
        confidence_list = []
        key_list = []

       
        for key, value in anno.items():
            
            feat_path_list.append(os.path.join(self.feature_path,key)+'.npy')
            
            label_list.append(value['score']-1)
            confidence_list.append(os.path.join(self.confidence_path,key)+'.npy')
            key_list.append(key)
            key_set = set(key_list)
        
        return feat_path_list, label_list, confidence_list, key_list

    def __len__(self):
        return len(self.feature_path_list)
    

    def delta(self):
        
        delta = []
        dataset = self.label_list.copy()
        for i in range(len(dataset)):
            for j in range(i+1,len(dataset)):
                delta.append(
                    abs(
                        self.label_list[i] -
                        self.label_list[j]))

        return delta


    def __getitem__(self, item):

        feat = torch.from_numpy(np.load(self.feature_path_list[item])).squeeze(dim=-1)
        conf = torch.from_numpy(np.load(self.confidence_list[item])).squeeze(dim=-1)[:feat.shape[1]]
       
        label = self.label_list[item]
       
        feat = F.interpolate(feat.unsqueeze(0),size=1000,mode='linear',align_corners=False)
        conf = F.interpolate(conf.unsqueeze(0).unsqueeze(0),size=1000,mode='linear',align_corners=False)
        feat = feat.squeeze()
        conf = conf.squeeze()

        if self.subset == 'test':
            train_file_list =[i for i in range(len(self.choose_list))] 
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]
            feat_list = []
            label_list = []
            conf_list= []
            for i in choosen_sample_list:
                feat_exampler = torch.from_numpy(np.load(self.feature_path_list[i])).squeeze(dim=-1)
                feat_exampler = F.interpolate(feat_exampler.unsqueeze(0),size=1000,mode='linear',align_corners=False)
                feat_exampler = feat_exampler.squeeze()
                conf_exampler = torch.from_numpy(np.load(self.confidence_list[i])).squeeze(dim=-1)
                conf_exampler = F.interpolate(conf_exampler.unsqueeze(0).unsqueeze(0),size=1000,mode='linear',align_corners=False)
                conf_exampler = conf_exampler.squeeze()
                
                label_exampler = self.label_list[i]
                feat_list.append(feat_exampler)
                conf_list.append(conf_exampler)
                label_list.append(label_exampler)
                
            return feat ,label, conf, feat_list, label_list, conf_list
        else:
            idx = random.randint(0, len((self.feature_path_list)) - 1)
            feat_exampler = torch.from_numpy(np.load(self.feature_path_list[idx])).squeeze(dim=-1)
            feat_exampler = F.interpolate(feat_exampler.unsqueeze(0),size=1000,mode='linear',align_corners=False)
            feat_exampler = feat_exampler.squeeze()
            label_exampler = self.label_list[idx]
            conf_exampler = torch.from_numpy(np.load(self.confidence_list[idx])).squeeze(dim=-1)
            conf_exampler = F.interpolate(conf_exampler.unsqueeze(0).unsqueeze(0),size=1000,mode='linear',align_corners=False)
            conf_exampler = conf_exampler.squeeze()
           
            return  feat, label, self.feature_path_list[item], conf, feat_exampler,label_exampler, self.feature_path_list[idx], conf_exampler
      