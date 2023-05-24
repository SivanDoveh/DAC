import ast
import json
import logging
import os
import sys
from dataclasses import dataclass
from multiprocessing import Value
import random
import braceexpand
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import (
    Dataset,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)
from SVLC_learning.negs_and_pos import (
    Negatives,
    Negatives_Auto,
    ChunkSample,
    RandBothNegatives,
    UseExtra,
    SAM_class,
    quality_caption_class,
)
import re
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import create_expansions.expanders_utils as expanders_utils
from open_clip import tokenize


def is_positive(attr, texts):
    positives_in_cap = [c for c in attr if c in texts]
    return len(positives_in_cap) > 0


def save_data(name, data, path):
    with open(f"{path}/{name}.json", "w") as f:
        json.dump(data, f)


def choose_negs_function(args):
    if args.neg_type == "auto":
        return Negatives_Auto(args)
    elif args.neg_type == "rand_both":
        return RandBothNegatives(args)
    else:
        return Negatives(args)


class CsvDataset(Dataset):
    def __init__(
        self, input_filename, transforms, img_key, caption_key, sep="\t", args=None
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)
        self.text_batch = args.mil_batch
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.root_dir = os.path.dirname(input_filename)
        self.args = args
        if self.args.use_only_quality_captions:
            images_with_blip_cap = os.listdir(args.quality_captions_folder)
            self.images = ["training/" + s.strip(".json") for s in images_with_blip_cap]
        if self.args.save_data and self.args.use_extra_cc3m_expanders:
            already_created = os.listdir(
                args.path_extra_cc3m_expanders_itm_scores_sentences
            )
            left = set([b.split("/")[-1] for b in self.images]) - set(
                [t.strip(".npy") for t in already_created]
            )
            self.images = ["training/" + s for s in left]
        if self.args.mil_dense:
            images_with_mil = os.listdir(self.args.mil_dense)
            images_with_mil_full_name = [
                "training/" + s.strip(".json") for s in images_with_mil
            ]
            common = list(set(self.images).intersection(images_with_mil_full_name))
            self.images = common
        self.QC_gen = quality_caption_class() if args.create_quality_captions else None
        self.SAM_gen = SAM_class(args) if args.create_SAM else None
        self.negs = choose_negs_function(args)
        self.use_extra = UseExtra(args)
        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info_dict = {}
        img_path = str(self.images[idx])
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root_dir, img_path)
        try:
            raw = Image.open(img_path)
            images = self.transforms(raw)
        except Exception as err:
            raw = Image.new("RGB", (256, 256))
            images = self.transforms(raw)

        if self.args.use_only_quality_captions:
            self.captions[idx] = json.load(
                open(
                    f'{self.args.quality_captions_folder}{img_path.split("/")[-1]}.json'
                )
            )["positive_caption"][0]
        if self.args.save_data:
            self.captions[idx] = self.captions[idx].translate(
                str.maketrans("", "", ",!.;#@-%^&*()?\/")
            )

        texts = tokenize([str(self.captions[idx])])[0]

        if self.args.save_data:
            if self.args.create_quality_captions:
                dict = self.QC_gen.create_cap(raw)
                if dict != "invalid":
                    save_data(
                        img_path.split("/")[-1],
                        dict,
                        self.args.quality_captions_folder,
                    )

            if self.args.create_SAM:
                if not os.path.isfile(
                    f'{self.args.SAM_dense_folder}/{img_path.split("/")[-1]}.json'
                ):
                    dict = self.SAM_gen.create_cap(raw)
                    if dict != "invalid":
                        save_data(
                            img_path.split("/")[-1], dict, self.args.SAM_dense_folder
                        )

        if self.args.vl_negs:
            negatives = self.negs.create_negs(self.captions[idx])
            info_dict.update({"negatives": negatives})
        if self.args.mil_dense:
            img_filename = os.path.basename(img_path)
            try:
                f = open(f"{os.path.join(self.args.mil_dense, img_filename)}.json")
                sentences_list_text = json.load(f)
                sentences_list_text = (
                    sentences_list_text["caption"]
                    if "SAM" in self.args.mil_dense
                    else sentences_list_text
                )
            except:
                print(
                    f"problem with reading {os.path.join(self.args.mil_dense, img_filename)}"
                )
                sentences_list_text = [self.captions[idx]]

            sentences_list_text.append(self.captions[idx])
            sentences_list_text = random.choices(sentences_list_text, k=self.text_batch)
            sentences_list = tokenize(sentences_list_text)
            if self.args.mil_dense_negs:
                sentences_list_negs = []
                for sen in sentences_list_text:
                    neg = self.negs.create_negs(sen)
                    sentences_list_negs.append(neg)
                if len(sentences_list_text) > self.text_batch:
                    sentences_list_negs = sentences_list_negs[-self.text_batch :]
                sentences_list_negs = torch.cat(sentences_list_negs)
                if sentences_list_negs.shape[0] < self.text_batch:
                    texts_num = sentences_list_negs.shape[0]
                    for _ in range(self.text_batch - texts_num):
                        pad = sentences_list_negs[
                            np.random.randint(texts_num), :
                        ].unsqueeze(0)
                        sentences_list_negs = torch.cat(
                            (sentences_list_negs, pad), dim=0
                        )
                sentences_list = torch.cat((sentences_list, sentences_list_negs))

        if self.args.save_data:
            return [], [], []

        if self.args.vl_pos or self.args.vl_negs:
            if self.args.mil_dense:
                return images, texts, info_dict, sentences_list
            else:
                return images, texts, info_dict
        elif self.args.mil_dense:
            return images, texts, sentences_list
        else:
            return images, texts

    @classmethod
    def collate_fn_mil_dense(self):
        # noinspection PyUnreachableCode
        def fun(data):
            img, text_inputs, mil_texts = zip(*data)
            img = torch.stack(img, 0)
            text_inputs = torch.stack(text_inputs, 0)
            mil_texts = torch.cat(mil_texts, 0)
            return img, text_inputs, mil_texts

        return fun

    @classmethod
    def collate_fn_mil_dense_plus_pos_or_neg(self):
        # noinspection PyUnreachableCode
        def fun(data):
            img, text_inputs, info_dict, mil_texts = zip(*data)
            img = torch.stack(img, 0)
            text_inputs = torch.stack(text_inputs, 0)
            info_dict = {
                "negatives": torch.stack(
                    [i_dict["negatives"] for i_dict in info_dict], 0
                )
            }
            mil_texts = torch.cat(mil_texts, 0)
            return img, text_inputs, info_dict, mil_texts

        return fun

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp((x - np.max(x)) * 10)
        return e_x / e_x.sum(axis=0)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

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


# def preprocess_neg(text):
#     negatives = negs.create_negs(text)
#     return tokenize([str(negatives)])[0]


# def pos_loading(sample):
#     try:
#         sample["pos"] = json.load(open(f'{LAION_path}{sample["__key__"]}.json'))
#     except:
#         sample["pos"] = str(sample["text"])
#     return sample


# def preprocess_pos(text):
#     return tokenize([str(text)])[0]


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


# def get_imagenet(args, preprocess_fns, split):
#     assert split in ["train", "val", "v2"]
#     is_train = split == "train"
#     preprocess_train, preprocess_val = preprocess_fns
#
#     if split == "v2":
#         from imagenetv2_pytorch import ImageNetV2Dataset
#
#         dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
#     else:
#         if is_train:
#             data_path = args.imagenet_train
#             preprocess_fn = preprocess_train
#         else:
#             data_path = args.imagenet_val
#             preprocess_fn = preprocess_val
#         # assert data_path
#
#         dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)
#
#     if is_train:
#         idxs = np.zeros(len(dataset.targets))
#         target_array = np.array(dataset.targets)
#         k = 50
#         for c in range(1000):
#             m = target_array == c
#             n = len(idxs[m])
#             arr = np.zeros(n)
#             arr[:k] = 1
#             np.random.shuffle(arr)
#             idxs[m] = arr
#
#         idxs = idxs.astype("int")
#         sampler = SubsetRandomSampler(np.where(idxs)[0])
#     else:
#         sampler = None
#
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         num_workers=args.workers,
#         sampler=sampler,
#     )
#
#     return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    return ("txt" in sample) and ("png" in sample or "jpg" in sample)


def save_pos_filter_no_caption_or_no_image(sample):
    if ("txt" in sample) and ("png" in sample or "jpg" in sample):
        regex = re.compile("[^a-zA-Z]")
        text = regex.sub(" ", str(sample["txt"]))
        text_name = sample["__key__"]
        if not os.path.isfile(f"{LAION_path}{text_name}.json"):
            positive = poss.create_pos(text)
            save_data(text_name, positive, f"{LAION_path}")
    return ("txt" in sample) and ("png" in sample or "jpg" in sample)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
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
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
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


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        args=args,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    if args.chunks > 1 and args.save_data:
        sampler = ChunkSample(
            dataset, shuffle=False, rank=args.curr_chunk, num_replicas=args.chunks
        )
    shuffle = is_train and sampler is None

    if args.mil_dense:
        if args.vl_pos or args.vl_negs:
            collate_fun = dataset.collate_fn_mil_dense_plus_pos_or_neg()
        else:
            collate_fun = dataset.collate_fn_mil_dense()
        dataloader = DataLoader(
            dataset,
            collate_fn=collate_fun,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
        )
    else:
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


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split(".")[-1]
        if ext in ["csv", "tsv"]:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}."
            )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False
        )

    if args.use_expanders_as_additional_data:
        data["train"] = get_2_datasets_expanders(args, preprocess_fns, is_train=True)

    return data


def get_2_datasets_expanders(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    # create two datasets with different sizes
    dataset1 = CsvDataset(
        input_filename,
        preprocess_fn[0],
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        args=args,
    )

    dataset2 = expanders_utils.Expander(
        args=args,
        transform=preprocess_fn[0],
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        input_filename=input_filename,
        sep=args.csv_separator,
    )

    if args.mil_co_loader:
        datasets = [dataset1, dataset2]
        dataloaders = []
        for dataset in datasets:
            sampler = (
                DistributedSampler(dataset) if args.distributed and is_train else None
            )
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
            sampler = ChunkSample(
                dataset, shuffle=False, rank=args.curr_chunk, num_replicas=args.chunks
            )

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
