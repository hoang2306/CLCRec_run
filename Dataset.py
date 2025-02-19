import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def data_load(args, exp_mode, dataset, has_v=True, has_a=True, has_t=True):
    dir_str = './Data/' + dataset
    train_data = np.load(dir_str+'/train.npy', allow_pickle=True)
    
    val_cold_data = np.load(dir_str+'/val_cold.npy', allow_pickle=True)
    
    test_cold_data = np.load(dir_str+'/test_cold.npy', allow_pickle=True)

    train_data = np.array(train_data, dtype=np.int32)

    # val_cold_data = np.array(val_cold_data, dtype=np.int32)
    # test_cold_data = np.array(test_cold_data, dtype=np.int32)

    if args.data_path == 'movielens':
        test_data = np.load(dir_str+'/test_full.npy', allow_pickle=True)
        test_warm_data = np.load(dir_str+'/test_warm.npy', allow_pickle=True)
        val_data = np.load(dir_str+'/val_full.npy', allow_pickle=True)
        val_warm_data = np.load(dir_str+'/val_warm.npy', allow_pickle=True)
        
    
    if dataset == 'movielens':
        num_user = 55485
        num_item = 5986
        num_warm_item = 5119
        # v_feat = torch.tensor(np.load(dir_str+'/feat_v.npy', allow_pickle=True), dtype=torch.float).cuda()
        a_feat = torch.tensor(np.load(dir_str+'/feat_a.npy', allow_pickle=True), dtype=torch.float).cuda()
        t_feat = torch.tensor(np.load(dir_str+'/feat_t.npy', allow_pickle=True), dtype=torch.float).cuda()

        v_feat = None  


    elif dataset == 'amazon':
        num_user = 27044
        num_item = 86506
        num_warm_item = 68810
        v_feat = torch.load(dir_str+'/feat_v.pt')
        a_feat = None
        t_feat = None

    elif dataset == 'tiktok':
        num_user = 32309
        num_item = 57832+8624
        num_warm_item = 57832
        if has_v:
            v_feat = torch.load(dir_str+'/feat_v.pt')
            v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        else:
            v_feat = None

        if has_a:
            a_feat = torch.load(dir_str+'/feat_a.pt')
            a_feat = torch.tensor(a_feat, dtype=torch.float).cuda() 
        else:
            a_feat = None
        
        t_feat = torch.load(dir_str+'/feat_t.pt').cuda()
    elif dataset == 'kwai':
        num_user = 7010
        num_item = 86483
        num_warm_item = 74470

        v_feat = np.load(dir_str+'/feat_v.npy')
        v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        a_feat = t_feat = None
    else:
        if dataset == 'baby':
            num_user = 19445
            num_item = 7050
            num_warm_item = 5640
        # add more 

        # multimedia load 
        # if self.env.args.exp_mode=='ff':
        #         image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
        #         text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')
        #         audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat.npy')
        #     elif self.env.args.exp_mode=='fm':
        #         image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_test.npy')
        #         text_file = os.path.join(self.env.DATA_PATH, 'text_feat_missing_test.npy')
        #         audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat_missing_test.npy')
        #     elif self.env.args.exp_mode=='mf':
        #         image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_train.npy')
        #         text_file = os.path.join(self.env.DATA_PATH, 'text_feat_missing_train.npy')
        #         audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat_missing_train.npy')
        #     elif self.env.args.exp_mode=='mm':
        #         image_file = os.path.join(self.env.DATA_PATH, 'image_feat_missing_all.npy')
        #         text_file = os.path.join(self.env.DATA_PATH, 'text_feat_missing_all.npy')
        #         audio_file = os.path.join(self.env.DATA_PATH, 'audio_feat_missing_all.npy')

        if exp_mode == 'ff':
            v_feat = os.path.join(dir_str, 'image_feat.npy')
            t_feat = os.path.join(dir_str, 'text_feat.npy')
            a_feat = os.path.join(dir_str, 'audio_feat.npy')
        elif exp_mode == 'fm':
            print('load multimedia ok')
            v_feat = os.path.join(dir_str, 'image_feat_missing_test.npy')
            t_feat = os.path.join(dir_str, 'text_feat_missing_test.npy')
            a_feat = os.path.join(dir_str, 'audio_feat_missing_test.npy')
        elif exp_mode == 'mf':
            v_feat = os.path.join(dir_str, 'image_feat_missing_train.npy')
            t_feat = os.path.join(dir_str, 'text_feat_missing_train.npy')
            a_feat = os.path.join(dir_str, 'audio_feat_missing_train.npy')
        elif exp_mode == 'mm':
            v_feat = os.path.join(dir_str, 'image_feat_missing_all.npy')
            t_feat = os.path.join(dir_str, 'text_feat_missing_all.npy')
            a_feat = os.path.join(dir_str, 'audio_feat_missing_all.npy')

        v_feat = np.load(v_feat)
        v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        t_feat = np.load(t_feat)
        t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()
        # not exist audio feature 
        a_feat = None 
    if args.data_path == 'movielens':
        return num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat
    else:
        return num_user, num_item, num_warm_item, train_data, val_cold_data, test_cold_data, v_feat, a_feat, t_feat

class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, dataset, train_data, num_neg):
        self.train_data = train_data
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict
        self.cold_set = set(np.load('./Data/'+dataset+'/cold_set.npy'))
        self.all_set = set(range(num_user, num_user+num_item))-self.cold_set

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, pos_item = self.train_data[index]
        neg_item = random.sample(self.all_set-set(self.user_item_dict[user]), self.num_neg)
        user_tensor = torch.LongTensor([user]*(self.num_neg+1))
        item_tensor = torch.LongTensor([pos_item] + neg_item)
        return user_tensor, item_tensor

