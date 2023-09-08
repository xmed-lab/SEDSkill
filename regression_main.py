from mvp_dataset import MVPDataset

import torch
import os
import torch.nn as nn
from model.TCN import SingleStageTCN_Con_Loc

from model.relative import TransformerDecoderLayer

from torch.utils.data import Dataset, DataLoader
import config
from scipy import stats
import numpy as np
CE = nn.CrossEntropyLoss()
MSE  = torch.nn.MSELoss(reduce=True, size_average=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_stages = 3  # refinement stages
num_layers = 10 # layers of prediction tcn e
num_f_maps = 64
dim = 2048
num_layers_PG = 11
num_layers_R = 10
num_R = 3
learning_rate = 1e-3
full_epoch = 200
num_classes = 9
fixed_length = 2000
f_path = os.path.abspath('..')
root_path = f_path.split('code')[0]


args = config.args
save_path ='./experiments/{}/{}'.format(args.dataset, args.save)
if args.pre_train:
        save_path = save_path + args.pre_train_path

if not os.path.exists(save_path): os.makedirs(save_path)

dataset_pth = root_path+'datasets/71_heart/'
train_anno_path = dataset_pth + 'annotations/train.json'
test_anno_path = dataset_pth + 'annotations/test.json'
feature_path = dataset_pth + 'features/videos/sample_rate25'
confidence_path = dataset_pth + 'features/confidence_score'

  
def train_con(model, relativer,train_loader, test_loader, device, save_dir):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
    best_acc = 10000
    best_epoch = 0
    best_rho = 0
    for epoch in range(1, full_epoch):
        mae = 0
        total = 0
        loss_item = 0
        for (video, labels,  _, conf, video_exampler, labels_exampler, _ ,conf_exampler) in (train_loader):
            video, labels= video.to(device), labels.to(device)
            conf, conf_exampler =  conf.to(device), conf_exampler.to(device)
            
            
            video_exampler, labels_exampler= video_exampler.to(device), labels_exampler.to(device)

            
            
            video = video.squeeze(dim=-1)
           
            video_exampler = video_exampler.squeeze(dim=-1)
            
            conf = conf.unsqueeze(dim=1)
            conf_exampler = conf_exampler.unsqueeze(dim=1)
            local_video = conf*video
            local_example = conf_exampler*video_exampler

            output, output_exampler, rela_score =  model(video, video_exampler,local_video, local_example )
            
         

            labels = labels.unsqueeze(dim=-1)
            
            mse_loss = MSE(output.float(), labels.float())
            mse_loss1 = MSE(output_exampler.float(), labels_exampler.float())
            rel_loss = MSE(rela_score.float(), abs(labels-labels_exampler).float())
            all_loss = mse_loss + mse_loss1 + rel_loss
            optimizer.zero_grad()
            all_loss.backward()
            loss_item += all_loss.item()
            optimizer.step()
            mae +=torch.abs(output.float() - labels.float()).sum()
            total += labels.shape[0]
           

        print('Train Epoch {}: Train Acc {}, Loss {}, '.format(epoch, mae / total, loss_item /total))
        accuracy, rho = test_con(model, relativer, test_loader, epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
       
        if accuracy < best_acc:
            best_acc = accuracy
            best_rho = rho
            torch.save(model.state_dict(),save_dir+'best_{}_acc{}.pth'.format(epoch,best_acc))
        print('Best Acc {}, Best Rho {}'.format(best_acc, best_rho))


def test_con(model, relativer, test_loader, epoch):
    model.eval()
    mae = 0
    total = 0
    loss_item = 0
    with torch.no_grad():
        
        for (video, labels, conf, video_2_list, label_2_list, conf_2_list) in (test_loader):
            video, labels = video.to(device), labels.to(device)
            
            video = video.squeeze(dim=-1)
            video_exampler, conf_exampler = video_2_list[0], conf_2_list[0]

            video_exampler = video_exampler.squeeze(dim=-1)
            video_exampler= video_exampler.to(device)
            
            conf , conf_exampler = conf.to(device), conf_exampler.to(device)
            conf = conf.unsqueeze(dim=1)
            conf_exampler = conf_exampler.unsqueeze(dim=1)
            local_video = conf*video
            local_example = conf_exampler*video_exampler
           
            output,output_exampler,rela_score =  model(video, video_exampler, local_video, local_example)
         
            labels = labels.unsqueeze(dim=-1)
            mes_loss =  MSE(output,labels)
       
            loss_item  += mes_loss.item()
           
           
            output =  torch.round(output)
            mae +=torch.abs(output.float() - labels.float()).sum().item()
            pred = output.clone()
           
            gt = labels.clone()

            pred = pred.cpu().numpy()
            gt = gt.cpu().numpy()
            pred_scores = np.array(pred)
            true_scores = np.array(gt)
           
            rho, p = stats.spearmanr(true_scores, pred_scores)
           
            total += labels.shape[0]
          
        print('Test Epoch {}: Acc {}, Loss {}'.format(epoch, mae / total, loss_item /total))
        return mae / total, rho




base_model = SingleStageTCN_Con_Loc(num_layers, 2048, num_f_maps, num_classes)
relativer = TransformerDecoderLayer(64,8)
train_dataset = MVPDataset(feature_path, train_anno_path, confidence_path)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False)

test_dataset = MVPDataset(feature_path, test_anno_path, confidence_path, subset='test')
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=False)



train_con(base_model,relativer, train_dataloader, test_dataloader, device, save_path)
