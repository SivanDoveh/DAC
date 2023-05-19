import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value
import random

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from concept_learning.color_list import color_list
from concept_learning.action_list import action_list
from concept_learning.material_list import material_list
from concept_learning.size_list import size_list
from concept_learning.state_list import state_list
# from transformers import AutoTokenizer, AutoModel
from SVLC_learning.negs_and_pos import Negatives,Negatives_Auto, ChunkSample,RandBothNegatives,UseExtra#,CapAnything#Blip1Gen,,Noun_Pos_Auto,BlipGen,BloomGen, RandThreeNegatives, Negatives_Winoground
import json
import re
import time
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import training.synt_utils as synt_utils
import training.expanders_utils as expanders_utils
from torch.utils.data.sampler import Sampler
from open_clip import tokenize


def is_positive(attr, texts):
    positives_in_cap = [c for c in attr if c in texts]
    return len(positives_in_cap) > 0

def save_data(name, data, path):
    with open(f"{path}/{name}.json", 'w') as f:
        json.dump(data, f)

def choose_negs_function(args):
    if args.neg_type=='auto':
        return Negatives_Auto(args)
    elif args.neg_type=='rand_both':
        return RandBothNegatives(args)
    elif args.neg_type=='rand_three':
        return RandThreeNegatives(args)
    elif args.neg_type == 'winoground':
        return Negatives_Winoground(args)
    else:
        return Negatives(args)

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", args=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.text_batch = args.mil_batch
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.root_dir = os.path.dirname(input_filename)
        self.args=args
        if self.args.only_blip_cap_2:
            images_with_blip_cap = os.listdir(args.CC3M_blip2_chat_cap_folder)
            self.images = ['training/' + s.strip('.json') for s in images_with_blip_cap]
        if self.args.only_blip_cap_1:
            images_with_blip_cap = os.listdir(args.CC3M_blip1_cap_folder)
            self.images = ['training/' + s.strip('.json') for s in images_with_blip_cap]
        if self.args.save_data and self.args.use_extra_cc3m_expanders:
            already_created = os.listdir(args.path_extra_cc3m_expanders_itm_scores_sentences)
            left = set([b.split('/')[-1] for b in self.images])-set([t.strip('.npy') for t in already_created])
            self.images = ['training/' + s for s in left]
        if self.args.alpaca_expanders:
            images_with_alpaca = os.listdir(self.args.alpaca_expanders)
            self.images = ['training/' + s.strip('.txt') for s in images_with_alpaca]
        if self.args.mil_gpt:
            images_with_mil = os.listdir(self.args.mil_gpt)
            images_with_mil_full_name = ['training/' + s.strip('.json') for s in images_with_mil]
            common = list(set(self.images).intersection(images_with_mil_full_name))
            self.images = common
        # self.bloom_poss = BloomGen() if args.save_bloom_pos else None
        # self.bloom_neg = BloomGen() if args.vl_bloom_neg else None
        # self.create_pos_blip2 = BlipGen() if args.create_blip2_cap else None
        # self.create_pos_blip1 = Blip1Gen() if args.create_blip1_cap else None
        # self.noun_pos = Noun_Pos_Auto() if args.noun_pos else None
        # self.cap_anything_gen = CapAnything(args) if args.create_cap_anything else None
        self.negs = choose_negs_function(args)
        self.use_extra = UseExtra(args)
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)
        # return len(self.captions)

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

        if self.args.only_blip_cap_2:
            self.captions[idx] = json.load(open(f'{self.args.CC3M_blip2_chat_cap_folder}{img_path.split("/")[-1]}.json'))[
                        'positive_caption'][0]
        # if self.args.mil_gpt:
        #     # dictionary with image names as key and sentences list as value.
        #     if 'sivand' in args.mil_gpt:
        #         # text_list = open(os.path.join(args.mil_gpt, filename)).readlines()[0][1:-1].split('\\n')
        #         # text_list = [" ".join(text.replace('*', '').split()) for text in text_list if text]
        #         # key_sentences_dict[filename.split('.')[0]] = text_list
        #         with open(os.path.join(args.mil_gpt, 'key_sen.pkl'), 'rb') as data_file:
        #             key_sentences_dict = pickle.load(data_file)
        #     elif 'leonid' in args.mil_gpt:
        #         key_sentences_dict = {}
        #         folder = os.listdir(args.mil_gpt)
        #         if args.debug or args.fast_run:  # If in debug, only do that for 200 samples to save time.
        #             folder = folder[:200]
        #         for filename in folder:
        #             text_list = open(os.path.join(args.mil_gpt, filename)).readlines()
        #             text_list = [" ".join(text.replace('*', '').split()) for text in text_list if text]
        #             text_list = [text for text in text_list if len(text) > 5]
        #             key_sentences_dict[filename.split('.')[0]] = text_list
        #     elif 'alfassy' in args.mil_gpt:
        #         # text_list = json.load(open(os.path.join(args.mil_gpt, filename)))
        #         # key_sentences_dict[filename.split('.')[0]] = text_list
        #         with open(os.path.join(args.mil_gpt, 'key_sen.pkl'), 'rb') as data_file:
        #             key_sentences_dict = pickle.load(data_file)
        #     else:
        #         raise RuntimeError(f'{args.mil_gpt} path error.')
        #     self.key_sentences_dict = key_sentences_dict
        #     self.text_batch = args.mil_batch
        #     # We don't train on images which don't have gpt captions
        #     filtered_indices = [i for i, image in enumerate(self.images) if f'{image.split("/")[-1]}' in key_sentences_dict]
        #     filtered_images = [self.images[ind] for ind in filtered_indices]
        #     filtered_captions = [self.captions[ind] for ind in filtered_indices]
        #     # filtered_images_fast = [image for i, image in enumerate(self.images) if i in filtered_indices]
        #     # filtered_captions_fast = [caption for i, caption in enumerate(self.captions) if i in filtered_indices]
        #     self.images = filtered_images
        #     self.captions = filtered_captions
        #     print(f"Training on {len(self.images)} images with gpt captions.")

        if self.args.only_blip_cap_1:
            self.captions[idx] = json.load(open(f'{self.args.CC3M_blip1_cap_folder}{img_path.split("/")[-1]}.json'))[
                        'positive_caption'][0]
        if self.args.save_data:
            self.captions[idx] = self.captions[idx].translate(str.maketrans('','', ',!.;#@-%^&*()?\/'))

        texts = tokenize([str(self.captions[idx])])[0]

        if self.args.vl_pos:
            # try:
            if self.args.blip_cap:
                positive = json.load(open(f'{self.args.CC3M_blip2_chat_cap_folder}{img_path.split("/")[-1]}.json'))[
                        'positive_caption']
            elif self.args.noun_pos:
                positive =  self.noun_pos.create_noun_pos(self.captions[idx])
            elif self.args.use_extra_cc3m_expanders or self.args.use_extra_blip_cap_expanders or self.args.use_v2_extra_blip_expanders:
                if self.args.avg_pos_features:
                    sentences_list, match_list = self.use_extra.create_pos(img_path.split("/")[-1],self.captions[idx],raw)
                    padded_match_list = np.zeros(100)
                    padded_match_list[0:len(match_list)] = match_list
                    info_dict.update({"match_list": padded_match_list})

                    positive = sentences_list # afterwards in 'train' need to add- if args.avg...: split to tokenized sentences
                else:
                    positive = self.use_extra.create_pos(img_path.split("/")[-1],self.captions[idx],raw)
            else:
                positive = json.load(open(f'{self.args.CC3M_positivies_folder}{img_path.split("/")[-1]}.json'))['positive']

                # # #################### rebuttel
                # img = Image.open(img_path)
                # img.save(f"/dccstor/sivandov1/data/rebut2/{str(self.images[idx]).split('/')[-1]}.jpg")
                # file1 = open("/dccstor/sivandov1/data/rebut2/MyFile.txt", "a")
                # file1.write(f'image number:{str(self.images[idx]).split("/")[-1]}\noriginal caption: {self.captions[idx]}\npositive: {positive}\n')
                # cap = self.captions[idx]
                # negatives = self.negs.create_negs(self.captions[idx])
                # from open_clip.tokenizer import SimpleTokenizer
                # s = SimpleTokenizer()
                # ne=s.decode(negatives[0].cpu().numpy()).strip("!")
                # file1.write(f'negatives:{s.decode(negatives[0].cpu().numpy()).strip("!")}\n\n\n\n')
                # file1.close()
                # # #################### rebuttel ends
            # except:
            #     positive = self.captions[idx]
            if self.args.calc_pos_sim:
                info_dict.update({"positives_text": [positive]})
                info_dict.update({"text": [[str(self.captions[idx])]]})
            positive = tokenize(positive)
            if self.args.avg_pos_features:
                padded = torch.zeros([100, positive[0].shape[0]],dtype=torch.int64)
                padded[0:positive.shape[0]]=positive
                positive = padded
            info_dict.update({"positives": positive})

        if self.args.save_data:
            if self.args.save_bloom_pos:
                if not os.path.isfile(f'{self.args.CC3M_positivies_folder}{img_path.split("/")[-1]}.json'):
                    positive = self.bloom_poss.create_pos(self.captions[idx])
                    if positive != 'invalid':
                        save_data(img_path.split('/')[-1], positive, self.args.CC3M_positivies_folder)

            if self.args.save_bloom_neg:
                if not os.path.isfile(f'{self.args.CC3M_bloom_neg_folder}{img_path.split("/")[-1]}.json'):
                    generated = self.bloom_neg.create_bloom_neg(self.captions[idx])
                    if generated != 'invalid':
                        save_data(img_path.split('/')[-1], generated, self.args.CC3M_bloom_neg_folder)

            if self.args.create_blip2_cap:
                dict = self.create_pos_blip2.create_pos(raw)
                if dict != 'invalid':
                    save_data(img_path.split('/')[-1], dict, self.args.CC3M_blip2_chat_cap_folder)

            if self.args.create_cap_anything:
                if not os.path.isfile(f'{self.args.cap_anything_folder}/{img_path.split("/")[-1]}.json'):
                    dict = self.cap_anything_gen.create_cap(raw)
                    if dict != 'invalid':
                        save_data(img_path.split('/')[-1], dict, self.args.cap_anything_folder)

            if self.args.create_blip1_cap:
                dict = self.create_pos_blip1.create_pos(raw)
                if dict != 'invalid':
                    save_data(img_path.split('/')[-1], dict, self.args.CC3M_blip1_cap_folder)
            if self.args.alpaca_expanders:
                self.use_extra.create_sen(img_path.split("/")[-1])

        if self.args.vl_negs:
            if self.args.vl_bloom_neg:
                if not self.args.save_data:
                    try:
                        negatives = json.load(open(f'{self.args.CC3M_bloom_neg_folder}{img_path.split("/")[-1]}.json'))['negative']
                        negatives = tokenize(negatives)
                    except:
                        negatives = tokenize(['']) * 0

                    info_dict.update({"negatives": negatives})
            else:
                negatives = self.negs.create_negs(self.captions[idx])
                info_dict.update({"negatives": negatives})
        if self.args.mil_gpt:
            img_filename = os.path.basename(img_path)
            try:
                f = open(f'{os.path.join(self.args.mil_gpt, img_filename)}.json')
                sentences_list_text = json.load(f)
                sentences_list_text = sentences_list_text['caption'] if 'cap_anything' in self.args.mil_gpt else sentences_list_text
                if 'cap_anything' in self.args.mil_gpt:
                    phrases_to_remove = [
                        "the image shows a woman in a white dress",
                        "a person in a chair",
                    ]
                    filtered_sentences = []

                    for sentence in sentences_list_text:
                        if not any(phrase in sentence for phrase in phrases_to_remove):
                            filtered_sentences.append(sentence)
                    sentences_list_text = filtered_sentences
            except:
                print(f'problem with reading {os.path.join(self.args.mil_gpt, img_filename)}')
                sentences_list_text = [self.captions[idx]]

            sentences_list_text.append(self.captions[idx])
            sentences_list_text = random.choices(sentences_list_text, k=self.text_batch)
            # if len(sentences_list_text) > self.text_batch:
                # sentences_list_text = sentences_list_text[-self.text_batch:]
            sentences_list = tokenize(sentences_list_text)
            # if sentences_list.shape[0] < self.text_batch:
            #     texts_num = sentences_list.shape[0]
            #     for _ in range(self.text_batch - texts_num):
            #         pad = sentences_list[np.random.randint(texts_num), :].unsqueeze(0)
            #         sentences_list = torch.cat((sentences_list, pad), dim=0)
            if self.args.mil_gpt_negs:
                sentences_list_negs=[]
                for sen in sentences_list_text:
                    neg = self.negs.create_negs(sen)
                    sentences_list_negs.append(neg)
                if len(sentences_list_text) > self.text_batch:
                    sentences_list_negs = sentences_list_negs[-self.text_batch:]
                sentences_list_negs = torch.cat(sentences_list_negs)
                if sentences_list_negs.shape[0] < self.text_batch:
                    texts_num = sentences_list_negs.shape[0]
                    for _ in range(self.text_batch - texts_num):
                        pad = sentences_list_negs[np.random.randint(texts_num), :].unsqueeze(0)
                        sentences_list_negs = torch.cat((sentences_list_negs, pad), dim=0)

                sentences_list=torch.cat((sentences_list,sentences_list_negs))

            # sentences_list = torch.cat((texts.unsqueeze(0), sentences_list), dim=0)# add caption to mil


        if self.args.save_data:
            return [],[],[]

        if self.args.vl_pos or self.args.vl_negs:
            if self.args.mil_gpt:
                return images, texts, info_dict, sentences_list
            else:
                return images, texts, info_dict
        elif self.args.mil_gpt:
            return images, texts, sentences_list
        else:
            return images, texts

    @classmethod
    def collate_fn_mil_gpt(self):
        # noinspection PyUnreachableCode
        def fun(data):
            img, text_inputs, mil_texts = zip(*data)
            img = torch.stack(img, 0)
            text_inputs = torch.stack(text_inputs, 0)
            mil_texts = torch.cat(mil_texts, 0)
            return img, text_inputs, mil_texts
        return fun

    @classmethod
    def collate_fn_mil_gpt_plus_pos_or_neg(self):
        # noinspection PyUnreachableCode
        def fun(data):
            img, text_inputs, info_dict, mil_texts = zip(*data)
            img = torch.stack(img, 0)
            text_inputs = torch.stack(text_inputs, 0)
            info_dict = {'negatives': torch.stack([i_dict['negatives'] for i_dict in info_dict], 0)}
            mil_texts = torch.cat(mil_texts, 0)
            return img, text_inputs, info_dict, mil_texts
        return fun

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp((x - np.max(x))*10)
        return e_x / e_x.sum(axis=0)

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_txt(text):
    return tokenize([str(text)])[0]

def preprocess_neg(text):
    negatives = negs.create_negs(text)
    return tokenize([str(negatives)])[0]
def pos_loading(sample):
    try:
        sample["pos"] = json.load(open(f'{LAION_path}{sample["__key__"]}.json'))
    except:
        sample["pos"] = str(sample['text'])
    return sample

def preprocess_pos(text):
    return tokenize([str(text)])[0]


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        # assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    return ('txt' in sample) and ('png' in sample or 'jpg' in sample)

def save_pos_filter_no_caption_or_no_image(sample):
    if ('txt' in sample) and ('png' in sample or 'jpg' in sample):
        regex = re.compile('[^a-zA-Z]')
        text = regex.sub(' ', str(sample['txt']))
        text_name =sample['__key__']
        if not os.path.isfile(f'{LAION_path}{text_name}.json'):
            positive = poss.create_pos(text)
            save_data(text_name, positive, f'{LAION_path}')
    return ('txt' in sample) and ('png' in sample or 'jpg' in sample)

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False):
    global LAION_path
    LAION_path = args.LAION_positivies_folder
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    global negs
    negs = choose_negs_function(args)#Negatives_Auto(args) if args.neg_auto else Negatives(args)
    global poss

    poss = BloomGen()
    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train and not args.save_data:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    if args.save_data:
        pipeline.extend([
            wds.select(save_pos_filter_no_caption_or_no_image),
        ])
    else:
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
        ])

    pipeline.extend([
        wds.decode("pilrgb", handler=log_and_continue),])
    if args.vl_pos or args.vl_negs:
        pipeline.extend([
            wds.rename(image="jpg;png", text="txt",neg="txt",pos="txt"), #info_dict="txt"
            wds.map(pos_loading),
            wds.map_dict(image=preprocess_img, text=preprocess_txt,neg=preprocess_neg,pos=preprocess_pos),
            wds.to_tuple("image", "text","neg","pos"),
            wds.batched(args.batch_size, partial=not is_train), ])
    else:#if not vl_pos or negs - negatives/positives will not be created and plain text will return. on the next stage, they will not be used because they only used if flags are true
        pipeline.extend([
            wds.rename(image="jpg;png", text="txt",neg="txt",pos="txt"),
            wds.map_dict(image=preprocess_img, text=preprocess_txt,neg=preprocess_txt,pos=preprocess_txt),
            wds.to_tuple("image", "text","neg","pos"),
            wds.batched(args.batch_size, partial=not is_train), ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)
    if args.workers==0:
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=False,
        )
    else:
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=True,
        )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    if args.mil_gpt:
        if args.vl_pos or args.vl_negs:
            collate_fun = dataset.collate_fn_mil_gpt_plus_pos_or_neg()
        else:
            collate_fun = dataset.collate_fn_mil_gpt()
        dataloader = DataLoader(dataset, collate_fn=collate_fun, batch_size=args.batch_size,
                                shuffle=shuffle, num_workers=args.workers, pin_memory=True, sampler=sampler,
                                drop_last=is_train)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                                pin_memory=True, sampler=sampler, drop_last=is_train)

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        args = args)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    if args.chunks>1 and args.save_data:
        sampler = ChunkSample(dataset, shuffle=False, rank=args.curr_chunk, num_replicas=args.chunks)
    shuffle = is_train and sampler is None

    if args.mil_gpt:
        if args.vl_pos or args.vl_negs:
            collate_fun = dataset.collate_fn_mil_gpt_plus_pos_or_neg()
        else:
            collate_fun = dataset.collate_fn_mil_gpt()
        dataloader = DataLoader(dataset, collate_fn=collate_fun, batch_size=args.batch_size,
                                shuffle=shuffle, num_workers=args.workers, pin_memory=True, sampler=sampler,
                                drop_last=is_train)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                                pin_memory=True, sampler=sampler, drop_last=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    if args.use_synt_data:
        data["train"] = get_2_datasets_synth(args, preprocess_fns, is_train=True)
    if args.use_expanders_as_additional_data:
        data["train"] = get_2_datasets_expanders(args, preprocess_fns, is_train=True)

    return data


def get_2_datasets_expanders(args,preprocess_fn, is_train):

    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    # create two datasets with different sizes
    dataset1 = CsvDataset(
            input_filename,
            preprocess_fn[0],
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            args=args)

    dataset2 = expanders_utils.Expander(args = args,transform = preprocess_fn[0],img_key=args.csv_img_key,caption_key=args.csv_caption_key,input_filename=input_filename,sep=args.csv_separator)

    if args.mil_co_loader:
        datasets = [dataset1,dataset2]
        dataloaders=[]
        for dataset in datasets:
            sampler = DistributedSampler(dataset) if args.distributed and is_train else None
            shuffle = is_train and sampler is None
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.workers,
                pin_memory=True,
                sampler=sampler,
            )
            dataloader.num_samples = len(dataset)
            dataloader.num_batches = len(dataloader)
            dataloaders.append(DataInfo(dataloader, sampler))
        return dataloaders
    else:
        numbers = list(range(len(dataset2)))
        chosen_numbers = random.choices(numbers, k=len(dataset1))
        # create new datasets with the same length by repeating samples from the smaller dataset
        dataset2 = torch.utils.data.dataset.Subset(dataset2, chosen_numbers)

        # concatenate the datasets into a single dataset
        dataset = ConcatDataset([dataset1, dataset2])

        # create a dataloader for the concatenated dataset

        num_samples = len(dataset)
        sampler = DistributedSampler(dataset) if args.distributed and is_train else None
        if args.chunks > 1 and args.save_data:
            sampler = ChunkSample(dataset, shuffle=False, rank=args.curr_chunk, num_replicas=args.chunks)

        shuffle = is_train and sampler is None

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,

        )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_2_datasets_synth(args,preprocess_fn, is_train):

    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    # create two datasets with different sizes
    dataset1 = CsvDataset(
            input_filename,
            preprocess_fn[0],
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            args=args)

    dataset2 = synt_utils.dataset_synt(args,preprocess_fn[0])

    # determine the maximum length of the datasets
    # max_len = max(len(dataset1), len(dataset2))
    numbers = list(range(len(dataset2)))  # Create a list of numbers from 0-9
    chosen_numbers = random.choices(numbers, k=len(dataset1))
    # create new datasets with the same length by repeating samples from the smaller dataset
    dataset2 = torch.utils.data.dataset.Subset(dataset2, chosen_numbers)

    # concatenate the datasets into a single dataset
    dataset = ConcatDataset([dataset1, dataset2])

    # create a dataloader for the concatenated dataset

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    if args.chunks > 1 and args.save_data:
        sampler = ChunkSample(dataset, shuffle=False, rank=args.curr_chunk, num_replicas=args.chunks)
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)





