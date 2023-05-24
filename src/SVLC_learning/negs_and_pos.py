from matplotlib import pyplot as plt

from SVLC_learning.color_list import color_list
from SVLC_learning.action_list import action_list
from SVLC_learning.material_list import material_list
from SVLC_learning.size_list import size_list
from SVLC_learning.state_list import state_list
import os
import sys
import math
import json
import logging
import functools
import random
import pdb
from torchvision import transforms
import time
import pandas as pd
import numpy as np
from PIL import Image
from typing import Union
from dataclasses import dataclass
import torch
from open_clip.tokenizer import tokenize
import sys
import requests

# from transformers import BloomTokenizerFast, BloomForCausalLM
from torch.utils.data import Sampler, Dataset
from transformers import pipeline
import spacy
import string
import torch

# from transformers import BloomTokenizerFast
from torch import distributed as dist
import requests
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
import nltk
import re

# from BLIP1.models.blip import blip_decoder
from torchvision.transforms.functional import InterpolationMode
import cv2


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def is_positive(attr, texts):
    positives_in_cap = [c for c in attr if c in texts]
    return len(positives_in_cap) > 0


class RandBothNegatives(object):
    def __init__(self, args) -> None:
        self.Negatives = Negatives(args)
        self.Negatives_Auto = Negatives_Auto(args)
        self.args = args

    def create_negs(self, caption):
        neg_type_curr = random.choice([0, 1])
        if neg_type_curr == 0:
            negatives = self.Negatives.create_negs(caption)
        else:
            negatives = self.Negatives_Auto.create_negs(caption)

        return negatives


class UseExtra(object):
    def __init__(self, args) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.args = args
        nltk.download("punkt")
        self.sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        self.count_no_positive = 0
        self.count_positive = 0
        (
            self.model,
            self.vis_processors,
            self.text_processors,
        ) = load_model_and_preprocess(
            "blip2_image_text_matching", "pretrain", device=self.device, is_eval=True
        )

        self.nlp = spacy.load("en_core_web_sm")
        self.path_text = self.return_path_text()
        self.path_sentences = self.return_path_sentences()
        self.itm_scores_sentences = (
            self.args.path_extra_cc3m_expanders_itm_scores_sentences
            if self.args.use_extra_cc3m_expanders
            else self.args.path_extra_blip_cap_expanders_itm_scores_sentences
        )
        self.path_sub_sen = self.return_path_sub_sentences()

    def return_path_text(self):
        if self.args.use_v2_extra_cc3m_expanders:
            return self.args.v2_path_extra_cc3m_expanders
        elif self.args.use_extra_cc3m_expanders:
            return self.args.path_extra_cc3m_expanders
        elif self.args.use_v2_extra_blip_expanders:
            return self.args.v2_path_extra_blip_cap_expanders
        else:
            return self.args.path_extra_blip_cap_expanders

    def return_path_sub_sentences(self):
        if self.args.use_v2_extra_blip_expanders_noun:
            return self.args.v2_path_extra_blip_expanders_noun
        elif self.args.use_v2_extra_blip_expanders_adj:
            return self.args.v2_path_extra_blip_expanders_adj
        else:
            return ""

    def return_path_sentences(self):
        if self.args.use_v2_extra_cc3m_expanders:
            return self.args.v2_path_extra_cc3m_expanders_sen
        elif self.args.use_extra_cc3m_expanders:
            return self.args.path_extra_cc3m_expanders_sen
        elif self.args.use_v2_extra_blip_expanders:
            return self.args.v2_path_extra_blip_cap_expanders_sen
        else:
            return self.args.path_extra_blip_cap_expanders_sen

    def create_sen(self, img_name):
        # is sentence list exist

        try:
            f = open(f"{os.path.join(self.path_sentences, img_name)}.json")
            sentences_list = json.load(f)
        except:  # if not - create and save
            try:
                # with open(f'{os.path.join(self.path_text, img_name)}.txt', 'r') as f:
                #     text_expander = f.read()
                prompt_replace1 = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
                prompt_replace11 = "### Instruction:"
                prompt_replace12 = "A: please describe what you might see in a picture of a scene that contains a Christmas tree, write six sentences in a list, and use complete sentences with all nouns and objects you are referring to"
                prompt_replace13 = "B: "
                prompt_replace14 = "1. In the center of the room, a majestic evergreen Christmas tree stands tall, adorned with twinkling lights and colorful ornaments."
                prompt_replace15 = "2. Delicate strands of tinsel gracefully drape the tree>s branches, adding a touch of shimmer to the festive display."
                prompt_replace16 = "3. An elegant star or angel graces the top of the tree, representing the Star of Bethlehem or the heavenly messengers present at Jesus> birth."
                prompt_replace17 = "4. Wrapped presents in various shapes and sizes are piled beneath the tree, their festive gift wrap and bows hinting at the surprises inside."
                prompt_replace18 = "5. A cozy fireplace crackles nearby, with stockings hung from the mantel, eagerly awaiting the arrival of Santa Claus."
                prompt_replace19 = "6. Lush green garlands and flickering candles decorate the mantel, enhancing the holiday atmosphere."
                prompt_replace20 = "A: please describe what you might see in a picture of a scene that contains"
                prompt_replace21 = ", write six sentences in a list, and use complete sentences with all nouns and objects you are referring to"
                prompt_replace22 = "B:"
                prompt_replace23 = "### Response:"

                prompt_replace3 = "please describe what you might see in a picture of a scene that contains: "
                prompt_replace4 = ". Write five sentences in a list that describes the scene. Use complete sentences with all nouns and objects you are referring to. "

                if self.args.alpaca_expanders:
                    f = open(f"{os.path.join(self.path_text, img_name)}.txt")
                    txt = "\n".join(f.readlines())
                    clean_output = (
                        txt.replace(prompt_replace1, "")
                        .replace(prompt_replace11, "")
                        .replace(prompt_replace12, "")
                        .replace(prompt_replace13, "")
                        .replace(prompt_replace14, "")
                        .replace(prompt_replace15, "")
                        .replace(prompt_replace16, "")
                        .replace(prompt_replace17, "")
                        .replace(prompt_replace18, "")
                        .replace(prompt_replace19, "")
                        .replace(prompt_replace20, "")
                        .replace("\n", "")
                        .replace(prompt_replace21, "")
                        .replace(prompt_replace22, "")
                        .replace(prompt_replace23, " ")
                        .replace(prompt_replace3, "")
                        .replace(prompt_replace4, " ")
                    )
                    # clean_output = clean_output.replace("\n", "").replace(prompt_replace21, "").replace(
                    #     prompt_replace22, "").replace(prompt_replace23, " ").replace(prompt_replace3, "").replace(
                    #     prompt_replace4, " ")

                    sentences = re.split(r"\s*\d+\)\s*", clean_output)
                    sentences_list = list(filter(None, sentences))

                if len(sentences_list) > 0:
                    with open(
                        f"{os.path.join(self.path_sentences, img_name)}.json", "w"
                    ) as f:
                        json.dump(sentences_list, f)
            except:
                print("img doesnt have expander")

        return []

    def create_pos(self, img_name, caption, raw_image):
        positive = []
        try:
            # is sentence list exist
            try:
                f = open(f"{os.path.join(self.path_sentences, img_name)}.json")
                sentences_list = json.load(f)
            except:  # if not - create and save
                with open(f"{os.path.join(self.path_text, img_name)}.txt", "r") as f:
                    text_expander = f.read()

                if self.args.use_v2_extra_blip_expanders:
                    curr_text_clean = re.sub(" +", " ", text_expander).replace("\n", "")
                    curr_text_clean = curr_text_clean.split("short:")[0]
                    sentences_list = self.sent_tokenizer.tokenize(curr_text_clean)
                    sentences_list = [item.split("\t")[1] for item in sentences_list]
                else:

                    text_expander_clean = (
                        re.sub(" +", " ", text_expander)
                        .translate(str.maketrans("", "", ',#*\/"'))
                        .replace("\n", "")
                    )
                    sentences_list = self.sent_tokenizer.tokenize(
                        text_expander_clean
                    )

                if len(sentences_list) > 0:
                    with open(
                        f"{os.path.join(self.path_sentences, img_name)}.json", "w"
                    ) as f:
                        json.dump(sentences_list, f)
            # is itm_scores_sentences exist
            if self.args.mil_co_loader:
                bs = self.args.mil_co_loader_batch
                sentences_list = random.choices(sentences_list, k=bs)
                return sentences_list

            if self.args.use_pre_calc_matching:
                if self.args.random_sentence:
                    match_list = np.zeros(len(sentences_list)) * 0 + 1 / len(
                        sentences_list
                    )

                rand_caption_ind = np.random.choice(
                    np.arange(match_list.__len__()), p=match_list
                )
                positive = sentences_list[rand_caption_ind]
                self.count_positive += 1

            if (
                self.args.use_v2_extra_blip_expanders_noun
                or self.args.use_v2_extra_blip_expanders_adj
            ):
                try:
                    f = open(f"{os.path.join(self.path_sub_sen, img_name)}.json")
                    sub_sentences_list = json.load(f)
                except:
                    sub_sentences_list = []
                    temp_list = [self.nlp(s) for s in sentences_list]
                    for t in temp_list:
                        if self.args.use_v2_extra_blip_expanders_noun:
                            sub_sentences_list.extend(
                                [token.text for token in t if token.pos_ in ["NOUN"]]
                            )
                        else:
                            sub_sentences_list.extend(
                                [
                                    token.text
                                    for token in t
                                    if token.pos_ in ["VERB", "ADP", "ADJ", "PROPN"]
                                ]
                            )

                    if len(sub_sentences_list) > 0:
                        with open(
                            f"{os.path.join(self.path_sub_sen, img_name)}.json", "w"
                        ) as f:
                            json.dump(sub_sentences_list, f)

        except:
            positive = caption
            self.count_no_positive += 1
            if self.args.avg_pos_features:
                return positive, np.array([1])

        if self.args.avg_pos_features:
            return sentences_list, match_list

        return positive


class Negatives(object):
    def __init__(self, args) -> None:
        self.dict_lists = {
            "color": color_list,
            "action": action_list,
            "size": size_list,
            "state": state_list,
            "material": material_list,
        }
        self.args = args

    def create_negs(self, caption):
        # choose one random attribute that exist in the
        negs = len(self.args.vl_neg_type) * [0]
        for ind, neg_type in enumerate(self.args.vl_neg_type):
            negs[ind] = int(
                is_positive(self.dict_lists[neg_type], caption)
            )  # what attributes are in the text
        negs_possible_types = np.nonzero(negs)[
            0
        ]  # what are the possible attributes types
        selected_type = (
            random.choice(negs_possible_types) if len(negs_possible_types) != 0 else 0
        )
        neg_type = self.args.vl_neg_type[
            selected_type
        ]  # choose random attribute from the possible ones to be the positive type

        attributes = self.dict_lists[neg_type]
        negatives = tokenize([""] * self.args.num_negs) * 0
        positives_in_cap = [c for c in attributes if c in caption.split()]
        if len(positives_in_cap) > 0:
            neg_attr_text = []
            negative_attributes = list(set(attributes) - set(positives_in_cap))
            for i in range(self.args.num_negs):
                negative_attr = negative_attributes[
                    random.randint(0, len(negative_attributes) - 1)
                ]
                negative_attributes = list(set(negative_attributes) - {negative_attr})
                attr_to_change = positives_in_cap[
                    random.randint(0, len(positives_in_cap) - 1)
                ]
                neg_attr_text.append(caption.replace(attr_to_change, negative_attr))

            negatives = tokenize(neg_attr_text)

        return negatives


class Negatives_Auto(object):
    def __init__(self, args) -> None:
        self.classifier = pipeline("fill-mask")
        self.args = args
        self.nlp = spacy.load("en_core_web_sm")

    def create_negs(self, caption):
        if len(caption) > 512:
            caption = caption[:100]
        negatives = tokenize([""] * self.args.num_negs) * 0
        clean_caption = " ".join(
            caption.translate(str.maketrans("", "", string.punctuation)).split()
        )
        doc = self.nlp(clean_caption)
        # Analyze syntax
        positives_in_cap = [
            token.text for token in doc if token.pos_ in self.args.auto_neg_types
        ]
        positives_in_cap = list(
            set(positives_in_cap) - (set(positives_in_cap) - set(caption.split()))
        )  # filter one letter and weird mistakes of spacy
        if len(positives_in_cap) > 0:
            neg_attr_text = []
            for i in range(self.args.num_negs):
                attr_to_change = positives_in_cap[
                    random.randint(0, len(positives_in_cap) - 1)
                ]
                try:
                    pos_incides = np.nonzero(
                        [
                            1 if (w == attr_to_change) else 0
                            for w in clean_caption.split()
                        ]
                    )[0]
                    index_to_change = random.choice(pos_incides)
                    list_clean_cap = clean_caption.split()
                    list_clean_cap[index_to_change] = "<mask>"
                    fill_mask_list = self.classifier(" ".join(list_clean_cap))
                except:
                    print("fill-mask issue")

                try:
                    filttered_from_GT = [
                        item
                        for item in fill_mask_list
                        if not (item["token_str"].strip(" ") == attr_to_change)
                    ]
                    negative_caption = filttered_from_GT[
                        random.randint(0, len(filttered_from_GT) - 1)
                    ][
                        "sequence"
                    ]  # [-1]
                    neg_attr_text.append(negative_caption)
                    negatives = tokenize(neg_attr_text)
                except:
                    print("post_process negs issue")

        return negatives


class SAM_class(object):
    def __init__(self, args) -> None:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.args = args
        self.device = device
        from SAM.segmenter_folder import BaseSegmenter
        from SAM.captioner import build_captioner

        self.segmenter = BaseSegmenter(
            device="cuda",
            checkpoint=args.model_SAM,
            model_type="vit_h",
        )
        self.captioner = build_captioner("blip2", "cuda", args)

    def create_cap(self, raw_image):
        try:
            image = raw_image.convert("RGB")
            prompt = {"prompt_type": ["everything"]}
            # masks, anns = self.segmenter.inference(np.array(image), prompt)
            masks = self.segmenter.inference(np.array(image), prompt)
            # masks_sizes = np.sum(masks, axis=(1, 2))
            # top10_indices = np.argsort(masks_sizes)[::-1][:10]
            # masks = masks[top10_indices]
            ker_sz = 15
            sz_thresh = 0.01 * np.array(image.size).prod()
            captions_list = []
            for seg_mask in masks:
                if seg_mask.sum() < sz_thresh:
                    # print('skipping...')
                    continue
                seg_mask = 255 * seg_mask.astype(np.uint8)
                seg_mask = np.stack([seg_mask, seg_mask, seg_mask], axis=-1)
                seg_mask = cv2.morphologyEx(
                    seg_mask, cv2.MORPH_OPEN, kernel=np.ones((ker_sz, ker_sz), np.uint8)
                )
                seg_mask = cv2.morphologyEx(
                    seg_mask,
                    cv2.MORPH_CLOSE,
                    kernel=np.ones((ker_sz, ker_sz), np.uint8),
                )
                seg_mask = seg_mask[:, :, 0] > 0

                #  captioning with mask
                caption, crop_save_path = self.captioner.inference_seg(
                    image,
                    seg_mask,
                    crop_mode=self.args.seg_crop_mode,
                    filter=self.args.clip_filter,
                    disable_regular_box=self.args.disable_regular_box,
                )

                captions_list.append(caption)
        except:
            return "invalid"
        return {"caption": captions_list}

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann["segmentation"]
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))


class quality_caption_class(object):
    def __init__(self) -> None:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        model_caption, vis_processors_cap, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
        )
        self.device = device
        self.model_caption = model_caption
        self.vis_processors_cap = vis_processors_cap

    def create_cap(self, raw_image):
        try:
            image = (
                self.vis_processors_cap["eval"](raw_image).unsqueeze(0).to(self.device)
            )
            caption = self.model_caption.generate({"image": image})

        except:
            return "invalid"

        return {"positive_caption": caption}

class ChunkSample(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        self.num_replicas = num_replicas
        self.num_samples = None
        self.dataset = dataset
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        self.all_indices = range(len(self.dataset))
        self.shuffle = shuffle
        self.seed = seed
        self.indices = self.all_indices[self.rank :: self.num_replicas]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
