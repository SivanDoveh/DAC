import csv
import glob
import json
import os
import pickle
import random
from pathlib import Path
import torch
import numpy as np
import torch
from PIL import Image, ImageOps
from tdw.librarian import ModelLibrarian
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from open_clip import tokenize
from SVLC_learning.negs_and_pos import Negatives,Negatives_Auto,RandBothNegatives,UseExtra
import pandas as pd


def choose_negs_function(args):
    if args.neg_type=='auto':
        return Negatives_Auto(args)
    elif args.neg_type=='rand_both':
        return RandBothNegatives(args)
    else:
        return Negatives(args)

class Expander(Dataset):
    def __init__(
            self,
            input_filename,
            img_key,
            caption_key,
            transform=None,
            args=[],
            sep=[],
    ):
        df = pd.read_csv(input_filename, sep=sep)
        self.images = df[img_key].tolist()
        self.transforms = transform
        self.root_dir = os.path.dirname(input_filename)
        self.args=args
        lists=[]
        if self.args.only_blip_cap_2 and (self.args.use_v2_extra_blip_expanders or self.args.use_v2_extra_blip_expanders_noun or self.args.use_v2_extra_blip_expanders_adj):
            images_with_text = os.listdir(args.CC3M_blip2_chat_cap_folder)
            if self.args.use_v2_extra_blip_expanders:
                lists.append(os.listdir(args.v2_path_extra_blip_cap_expanders_sen))
            if self.args.use_v2_extra_blip_expanders_noun:
                lists.append(os.listdir(args.v2_path_extra_blip_expanders_sen_noun))
            if self.args.use_v2_extra_blip_expanders_adj:
                lists.append(os.listdir(args.v2_path_extra_blip_expanders_adj))
            for l in lists:
                images_with_text = list(set(images_with_text).intersection(l))
        elif self.args.only_blip_cap_2 and self.args.use_v2_extra_blip_expanders:
            list1 = os.listdir(args.CC3M_blip2_chat_cap_folder)
            list2 = os.listdir(args.v2_path_extra_blip_cap_expanders_sen)
            images_with_text = list(set(list1).intersection(list2))
        elif self.args.only_blip_cap_2:
            images_with_text = os.listdir(args.CC3M_blip2_chat_cap_folder)
        elif self.args.use_v2_extra_blip_expanders:
            images_with_text = os.listdir(args.v2_path_extra_blip_cap_expanders_sen)
        else:
            images_with_text = self.image

        self.images = ['training/' + s.strip('.json') for s in images_with_text]

        self.negs = choose_negs_function(args)
        self.args = args
        self.use_extra = UseExtra(args)
        self.captions = df[caption_key].tolist()



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info_dict={}
        img_path = str(self.images[idx])
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root_dir, img_path)
        try:
            raw = Image.open(img_path)
            images = self.transforms(raw)
        except Exception as err:
            raw = Image.new('RGB', (256, 256))
            images = self.transforms(raw)

        self.captions[idx] = self.use_extra.create_pos(img_path.split("/")[-1],self.captions[idx],raw)
        if self.args.mil_co_loader:
            texts = tokenize(self.captions[idx])
            padded = torch.zeros([100, texts[0].shape[0]], dtype=torch.int64)
            padded[0:texts.shape[0]] = texts
            texts = padded
        else:
            texts = tokenize([str(self.captions[idx])])[0]

        if self.args.vl_negs:
            if self.args.mil_co_loader: #TODO check this
                negatives =[]
                for c in self.captions[idx]:
                    negatives.extend(self.negs.create_negs(c))
                padded = torch.zeros([100, negatives[0].shape[0]], dtype=torch.int64)
                padded[0:negatives.shape[0]] = negatives
                negatives = padded
            else:
                negatives = self.negs.create_negs(self.captions[idx])
            info_dict.update({"negatives": negatives})

        if self.args.vl_negs:# and not self.args.save_data:
            return images, texts, info_dict
        else:
            return images, texts


def choose_feat_mil(args, image_features, text_features,logit_scale,list_amount_of_pos):
    concat_logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_image = divide_list(concat_logits_per_image, list_amount_of_pos)
    text_feature_divided_list = divide_list(text_features.t(), list_amount_of_pos)

    images_num = len(logits_per_image)
    mil_text_features_per_image =[]
    for img_ind in range(images_num):
        matchings_of_image = logits_per_image[img_ind][img_ind]
        if args.mil_co_loader_type == 'max':
            _,max_idx = torch.max(matchings_of_image, dim=0)
            mil_text_features_per_image.append(text_feature_divided_list[img_ind][:,max_idx])
        # elif args.mil_co_loader_type == 'avg':
        #     mil_text_features_per_image.append((sum(multiply_rows(text_feature_divided_list[img_ind].t(),matchings_of_image))/len(matchings_of_image)))


    return torch.stack(mil_text_features_per_image)

def divide_list(item_list, division_list):
    divided_lists = []
    index = 0
    for division in division_list:
        divided_lists.append(item_list[:,index:index+division])
        index += division
    return divided_lists

def multiply_rows(A, B):
    assert B.dim() == 1, "B must be a 1D tensor"
    assert A.size(0) == B.size(0), "A and B must have the same number of rows"
    B = B.unsqueeze(1)  # Add a new dimension to B to match A
    return torch.mul(A, B)