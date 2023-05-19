from torch.utils.data import Dataset, DataLoader
import os
import nltk
import pandas as pd
import torch
import re
import json
from SVLC_learning.paola_negs_and_pos import Negatives

class TextDataset(Dataset):
    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.args = args
        folder_path = args.synt_captions
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
        self.save_dir = args.save_dir
        self.negs =  Negatives(args)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        caption_key = self.file_paths[idx].split('/')[-1].strip('.txt').split('_')[-1]
        if os.path.exists(f'{self.save_dir}/{caption_key}.json'):
            print(f"The file {caption_key}.json exists in the folder.")
            return []
        with open(self.file_paths[idx]) as f:
            caption = f.readlines()[0]

        negative = self.negs.create_negs(caption)
        if negative is not None:
            with open(f'{self.save_dir}/{caption_key}.json', 'w') as f:
                json.dump(negative, f)

        return []

def get_dataloader(args):
    dataset = TextDataset(args)
    sampler = None
    if args.chunks>1 and args.save_data:
        sampler = ChunkSample(dataset, shuffle=False, rank=args.curr_chunk, num_replicas=args.chunks)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
    )
    return dataloader