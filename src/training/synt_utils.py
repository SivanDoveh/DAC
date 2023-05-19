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
from SVLC_learning.negs_and_pos import Negatives,Negatives_Auto,RandBothNegatives

IMAGENET_NAMES_PATH = "/dccstor/leonidka1/data/imagenet/ILSVRC/Annotations/CLS-LOC/imagenet_class_index.json"
IMAGENET_VIDEO_MAPPING_PATH = "/dccstor/paolac1/syn4vl/captioning/resources/json/syn4vl_imagenet_objects_mapping.json"
IMAGENET_DATA_PATH = "/dccstor/leonidka1/data/imagenet/ILSVRC/Data/CLS-LOC/train"

HMDB51_VIDEO_MAPPING_PATH = "/dccstor/leonidka1/khaled_space/syn4vl/captioning/resources/json/syn4vl_hmdb51_mapping.json"

def dataset_synt(args,preprocess):
    # load all datasets
    dict_all_dataset_types = {}

    dict_all_dataset_types["material"] = TDW_MATERIAL(
        transform=preprocess,args=args)

    dict_all_dataset_types["size"] = TDW_SIZE(
        transform=preprocess,args=args)

    dict_all_dataset_types["action"] = TDW_ACTION(
        sample_type="uniform_delayed",
        transform=preprocess,
        sample_fps=6,
        dataset_subset=1,
        dataset_types="amass_coverage",
        use_real=False,
        style_transfer=False,
        args=args
    )
    type_data = ["action", "material", "size"]
    tdw_dataset = ConcatDataset(
        [dict_all_dataset_types[t] for t in type_data]
    )
    return tdw_dataset

def imagenet_select_random(categories, imagenet_names):
    cat = np.random.choice(categories)
    cat_path = os.path.join(IMAGENET_DATA_PATH, cat)
    img_name = np.random.choice(os.listdir(cat_path))
    img_path = os.path.join(cat_path, img_name)

    img_txt = imagenet_names[cat]
    img_txt = "a photo of " + ("an " if img_txt[0] in "aieuo" else "a ") + img_txt.replace("_", " ")
    return img_path, img_txt

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


class LOAD_DATA:
    def __init__(self):
        all_imgs = []
        list_files = []
        list_text = []

        subset_tdw = pickle.load(
            open("/dccstor/paolac1/notebooks/position_samples_tdw.p", "rb")
        )
        material_subset_tdw = pickle.load(
            open("/dccstor/paolac1/notebooks/color_and_material_samples_tdw.p", "rb")
        )

        img_folders = glob.glob(
            "/dccstor/paolac1/notebooks/position_abstractscenes_1.2/*"
        )
        for folder_tmp in img_folders:
            all_imgs.extend(glob.glob(folder_tmp + "/*"))

        for img_path in all_imgs:
            type_pos = img_path.split("/")[1]
            filename = img_path.split("/")[-1]
            img_parts = filename.split("_")

            if type_pos == "left_center_right":
                left_item = img_parts[1]
                middle_item = img_parts[3]
                right_item = img_parts[5]

                list_files.append(img_path + "*-*0")
                list_text.append(
                    "the {} is on the left and the {} is on the right of the {}".format(
                        left_item, right_item, middle_item
                    )
                )
                list_files.append(img_path + "*-*1")
                list_text.append(
                    "the {} is on the left and the {} is on the right of the {}".format(
                        right_item, left_item, middle_item
                    )
                )

            if type_pos == "left_right":
                left_item = img_parts[1]
                right_item = img_parts[3]

                list_files.append(img_path + "*-*0")
                list_text.append(
                    "the {} is on the left to the {}".format(left_item, right_item)
                )
                list_files.append(img_path + "*-*0")
                list_text.append(
                    "the {} is on the right of the {}".format(right_item, left_item)
                )
                list_files.append(img_path + "*-*1")
                list_text.append(
                    "the {} is on the left to the {}".format(right_item, left_item)
                )
                list_files.append(img_path + "*-*1")
                list_text.append(
                    "the {} is on the right of the {}".format(left_item, right_item)
                )

            if type_pos == "up_down":
                if img_parts[2] == "up.png":
                    list_files.append(img_path + "*-*0")
                    list_text.append(
                        "the {} is on top of the table".format(img_parts[1])
                    )
                    list_files.append(img_path + "*-*0")
                    list_text.append("the {} is above the table".format(img_parts[1]))
                    list_files.append(img_path + "*-*0")
                    list_text.append("the table is below the {}".format(img_parts[1]))

                if img_parts[2] == "down.png":
                    list_files.append(img_path + "*-*1")
                    list_text.append("the {} is below the table".format(img_parts[1]))
                    list_files.append(img_path + "*-*1")
                    list_text.append(
                        "the {} is at the bottom of the table".format(img_parts[1])
                    )
                    list_files.append(img_path + "*-*1")
                    list_text.append("the table is above the {}".format(img_parts[1]))

        self.list_files = list_files
        self.list_text = list_text
        self.subset_tdw = subset_tdw


class TDW_AS_DATA(Dataset):
    def __init__(self, list_files, list_text, subset_tdw, transform=None, load_filepaths=False,args =[]):
        self.file_list = list_files
        self.text_list = list_text
        self.load_filepaths = load_filepaths
        self.negs = choose_negs_function(args)
        self.args = args

        for k, v in subset_tdw.items():
            for sentence in v[0]:
                self.file_list.append(k + "*-*0")
                self.text_list.append(sentence)
            for sentence in v[1]:
                self.file_list.append(k + "*-*1")
                self.text_list.append(sentence)

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.load_filepaths:
            return self.file_list[idx][:-4], self.text_list[idx]

        img_path = self.file_list[idx][:-4]
        img = Image.open(img_path).convert("RGB")
        if (self.file_list[idx][-1]) in {"1", 1}:
            img = ImageOps.mirror(img)

        if self.transform:
            img = self.transform(img)

        text = self.text_list[idx]
        text_tokenized = convert_to_tokens_list(text)[0][0]
        if self.args.vl_negs:
            info_dict = {}
            negatives = self.negs.create_negs(text)
            info_dict.update({"negatives": negatives})
            return img, text_tokenized, info_dict

        return img, text_tokenized

def convert_to_tokens_list(text):
    # process
    split_descriptions_per_sample = []
    # for d in text:
    d=text
    d = d.split("*-*")[0]
    split_d = [di.strip() for di in d.split(".") if di.strip()]
    split_d = [". ".join(split_d[di:di + 4]).strip() for di in range(0, len(split_d), 4)]
    split_descriptions_per_sample.append(split_d[:2])
    descriptions_per_sample = split_descriptions_per_sample

    #choose one random sentence
    try:
        l = descriptions_per_sample[0][0].split('.')
        if len(l) > 0:
            descriptions_per_sample = [random.choice(l)]
    except:
        print(descriptions_per_sample)

    descriptions_tokens_atts = []
    # for di in descriptions_per_sample:
    di = descriptions_per_sample
    d_tokens = tokenize(di)
    descriptions_tokens_atts.append(d_tokens)
    descriptions_tokens_atts = torch.nn.utils.rnn.pad_sequence(descriptions_tokens_atts, batch_first=True)

    text = descriptions_tokens_atts
    return text

class TDW_COLORED(Dataset):
    def __init__(
            self,
            data_path="/dccstor/paolac1/data/TDW_NEW/colored/base_colored_images_descriptions_v3.p",
            transform=None,
            load_filepaths=False,
            args=[]
    ):
        base_colored_images_descriptions = pickle.load(open(data_path, "rb"))

        self.file_list = list(base_colored_images_descriptions.keys())
        self.text_list = list(base_colored_images_descriptions.values())
        self.transform = transform
        self.load_filepaths = load_filepaths
        self.negs = choose_negs_function(args)
        self.args = args

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.load_filepaths:
            return self.file_list[idx], self.text_list[idx]
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if random.choice([True, False]):
            img = ImageOps.mirror(img)

        if self.transform:
            img = self.transform(img)

        text = self.text_list[idx]
        text_tokenized = convert_to_tokens_list(text)[0][0]
        if self.args.vl_negs:
            info_dict = {}
            negatives = self.negs.create_negs(text)
            info_dict.update({"negatives": negatives})
            return img, text_tokenized, info_dict
        return img, text_tokenized


class TDW_ACTION(Dataset):
    def __init__(
            self,
            teach_path="/dccstor/leonidka1/khaled_space/syn4vl/captioning/data_teach",
            amass_path="//dccstor/leonidka1/khaled_space/syn4vl/captioning/data_amass",
            amass_coverage_path="/dccstor/leonidka1/khaled_space/syn4vl/captioning/data_amass_coverage",
            teach_valid_path="/dccstor/leonidka1/khaled_space/syn4vl/captioning/resources/csv/invalid_videos_TEACH.csv",
            amass_valid_path="/dccstor/leonidka1/khaled_space/syn4vl/captioning/resources/csv/invalid_videos_AMASS.csv",
            amass_coverage_valid_path="/dccstor/leonidka1/khaled_space/syn4vl/captioning/resources/csv/invalid_videos_AMASS_Coverage.csv",
            captioning_model="grammar_full",
            dataset_types="all",  # "all" or a subset of ["teach", "amass", "amass_coverage"]
            sample_type="None",
            sample_fps=5,
            dataset_subset=1,
            transform=None,
            use_real=False,
            style_transfer=None,
            load_filepaths=False,
            args=[]
    ):
        self.negs = choose_negs_function(args)
        self.args = args
        self.load_filepaths = load_filepaths
        caption_prefix_len = len("In this scene, we can see ")
        if dataset_types == "all":
            dataset_types = {"teach", "amass", "amass_coverage"}
        elif isinstance(dataset_types, str):
            dataset_types = [dataset_types]

        validation_paths = set()
        self.video_caption_mapping = {}
        if "teach" in dataset_types:
            validation_paths.add(teach_valid_path)
            self.get_teach_dataset(teach_path)
        if "amass" in dataset_types:
            validation_paths.add(amass_valid_path)
            self.get_amass_dataset(amass_path)
        if "amass_coverage" in dataset_types:
            validation_paths.add(amass_coverage_valid_path)
            self.get_amass_dataset(amass_coverage_path, captions_dirname="captions_valid")

        file_list_all = []
        text_list_all = []

        videos_valid_frames = {}
        videos_imagenet_map = {}

        # Load HMDB51 video-img similarity mappings
        with open(HMDB51_VIDEO_MAPPING_PATH) as f:
            self.hmdb51_mapping = json.load(f)

        # Load ImageNet video-img similarity mappings
        with open(IMAGENET_VIDEO_MAPPING_PATH) as f:
            imagenet_mapping = json.load(f)
        with open(IMAGENET_NAMES_PATH) as f:
            imagenet_names = json.load(f)
        imagenet_names = {v[0]: v[1] for k, v in imagenet_names.items()}

        for valid_path in validation_paths:
            with open(valid_path) as f:
                reader = csv.reader(f)
                valid_index = next(reader).index("2")
                for line in reader:
                    path = line[0]
                    if path in imagenet_mapping:
                        videos_imagenet_map[path] = imagenet_mapping[path]
                    valid_frames = [str(i) for i in eval(line[valid_index])]
                    if valid_frames:
                        videos_valid_frames[path] = valid_frames
        del imagenet_mapping

        # Select sample of the dataset, if specified
        valid_videos = list(videos_valid_frames.keys())
        if dataset_subset != 1:
            sampled_videos = np.random.choice(valid_videos, int(dataset_subset * len(valid_videos)), replace=False)
            videos_valid_frames = {v: videos_valid_frames[v] for v in sampled_videos}
            videos_imagenet_map = {v: videos_imagenet_map[v] for v in sampled_videos}
        print(f"ACTION FRAMES: Retrieved {len(valid_videos)} valid videos. Sampled {len(videos_valid_frames)} videos.")

        for video_path, captions_path in tqdm(self.video_caption_mapping.items()):
            if video_path not in videos_valid_frames:
                continue
            with open(captions_path) as captions_f:
                captions = json.load(captions_f)

            for frame, frame_captions in captions.items():
                if (
                        (frame not in videos_valid_frames[video_path])
                        or (captioning_model not in frame_captions)
                        or len(frame_captions[captioning_model]) < caption_prefix_len
                ):
                    continue

                frame_cap = frame_captions[captioning_model].strip()
                frame_cap = frame_cap.split(".")
                sampled_sentences = [ci.strip() for ci in frame_cap if (np.random.random() < (7 / len(frame_cap)))]
                frame_cap = ". ".join(sampled_sentences).strip()

                img_idx = "0" * (4 - len(frame)) + frame
                img_path = os.path.join(video_path, f"img_{img_idx}.jpg")

                if use_real:
                    imgnet_path, imgnet_caption = imagenet_select_random(videos_imagenet_map[video_path],
                                                                         imagenet_names)
                    file_list_all.append(imgnet_path)
                    text_list_all.append(imgnet_caption)

                file_list_all.append(img_path)
                text_list_all.append(frame_cap)

        n = len(file_list_all)
        if sample_type == "random":
            indices = np.random.choice(
                np.arange(n), int(np.ceil(n * sample_fps / 30)), replace=False
            )
        elif sample_type == "uniform":
            indices = []
            skip_every = 30 // sample_fps
            if use_real:
                skip_every *= 2
            for ii in np.arange(0, n, skip_every):
                indices.append(ii)
                if use_real:
                    indices.append(ii + 1)
        elif sample_type == "uniform_delayed":
            indices = []
            skip_every = 30 // sample_fps
            if use_real:
                skip_every *= 2
            for ii in np.arange(20, n, skip_every):
                indices.append(ii - 1)
                if use_real:
                    indices.append(ii + 1)
        else:
            indices = np.arange(n)

        self.file_list = [file_list_all[i] for i in indices]
        self.text_list = [text_list_all[i] for i in indices]

        print(
            f"ACTION FRAMES: Retrived {len(file_list_all)} frames, Sampled {len(self.file_list)} frames"
        )

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.load_filepaths:
            return self.file_list[idx], self.text_list[idx]
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        text = self.text_list[idx]
        text = text.replace("first", "").replace("second", "").replace("third", "")

        text_tokenized = convert_to_tokens_list(text)[0][0]
        if self.args.vl_negs:
            info_dict = {}
            negatives = self.negs.create_negs(text)
            info_dict.update({"negatives": negatives})
            return img, text_tokenized, info_dict

        return img, text_tokenized

    def get_teach_dataset(self, dataset_path):
        for video_idx in os.listdir(dataset_path):
            video_path = Path(os.path.join(dataset_path, video_idx)).resolve()
            captions_path = os.path.join(
                dataset_path, "captions", f"captions_{video_idx}.json"
            )
            if not os.path.isfile(captions_path):
                continue
            self.video_caption_mapping[str(video_path)] = captions_path

    def get_amass_dataset(self, dataset_path, captions_dirname="captions"):
        for video_idx in os.listdir(dataset_path):
            video_path = Path(os.path.join(dataset_path, video_idx)).resolve()
            for view in os.listdir(video_path):
                if not view.startswith("c"):
                    continue
                view_path = os.path.join(video_path, view)
                captions_path = os.path.join(
                    dataset_path, captions_dirname, f"captions_{video_idx}_{view}.json"
                )
                if not os.path.isfile(captions_path):
                    continue
                self.video_caption_mapping[str(view_path)] = captions_path


class TDW_NEW_POSITIONS(Dataset):
    def __init__(
            self,
            data_path="/dccstor/paolac1/data/TDW_NEW/position/base_position_images_descriptions_v3.p",
            transform=None,
            args=[]
    ):
        self.negs = choose_negs_function(args)
        self.args = args
        base_colored_images_descriptions = pickle.load(open(data_path, "rb"))

        self.file_list = list(base_colored_images_descriptions.keys())
        self.text_list = list(base_colored_images_descriptions.values())
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        text = self.text_list[idx]
        text = text.split("*-*")

        text = convert_to_tokens_list(text[0])
        if self.args.vl_negs:
            info_dict = {}
            negatives = self.negs.create_negs(text)
            info_dict.update({"negatives": negatives})
            return img, text, info_dict
        return img, text[0]


class TDW_NEW_POSITIONS_AND_AS(Dataset):
    def __init__(
            self,
            list_files,
            list_text,
            data_path="/dccstor/paolac1/data/TDW_NEW/position/base_position_images_descriptions_v3.p",
            transform=None,
            args=[]
    ):
        self.negs = choose_negs_function(args)
        self.args = args
        base_colored_images_descriptions = pickle.load(open(data_path, "rb"))

        self.file_list = list(base_colored_images_descriptions.keys())
        self.text_list = list(base_colored_images_descriptions.values())

        self.file_list.extend(list_files)
        self.text_list.extend(list_text)

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        text = self.text_list[idx]
        text = text.split("*-*")

        text = convert_to_tokens_list(text[0])
        if self.args.vl_negs:
            info_dict = {}
            negatives = self.negs.create_negs(text)
            info_dict.update({"negatives": negatives})
            return img, text, info_dict
        return img, text[0]


class TDW_SIZE(Dataset):
    def __init__(
            self,
            data_path="/dccstor/paolac1/data/TDW_NEW/size/base_size_images_descriptions_v3.p",
            transform=None,
            load_filepaths=False,
            args=[]
    ):
        self.negs = choose_negs_function(args)
        self.args = args
        self.load_filepaths = load_filepaths
        base_colored_images_descriptions = pickle.load(open(data_path, "rb"))

        self.file_list = list(base_colored_images_descriptions.keys())
        self.text_list = list(base_colored_images_descriptions.values())

        self.transform = transform

        library = ModelLibrarian(library="models_full.json")
        self.all_tdw_records = {}
        for rec in library.records:
            self.all_tdw_records[rec.name] = rec.wcategory

        self.big = descriptors.BIG
        self.small = descriptors.SMALL
        self.comodin = descriptors.COMODIN

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        text = text.split("*-*")

        obj_right = self.all_tdw_records[text[1]]
        obj_left = self.all_tdw_records[text[2]]

        desc1 = f"a {random.choice(self.small)} {obj_right} and a {random.choice(self.big)} {obj_left}"  ## TT
        desc2 = f"{random.choice(self.big)} {obj_left}"
        desc3 = f"{random.choice(self.small)} {obj_right}"
        desc4 = random.choice(self.comodin)
        text = random.choice([desc1, desc2, desc3, desc4])

        if self.load_filepaths:
            return self.file_list[idx], text

        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")

        should_flip = random.choice([True, False])
        if should_flip:
            img = ImageOps.mirror(img)

        if self.transform:
            img = self.transform(img)

        text_tokenized = convert_to_tokens_list(text)[0][0]
        if self.args.vl_negs:
            info_dict = {}
            negatives = self.negs.create_negs(text)
            info_dict.update({"negatives": negatives})
            return img, text_tokenized, info_dict

        return img, text_tokenized


class TDW_MATERIAL(Dataset):
    def __init__(
            self,
            data_path="/dccstor/paolac1/data/TDW_NEW/materials/base_material_images_descriptions_v3.p",
            transform=None,
            load_filepaths=False,
            args=[]
    ):
        self.negs = choose_negs_function(args)
        self.args = args
        self.load_filepaths = load_filepaths
        base_colored_images_descriptions = pickle.load(open(data_path, "rb"))

        self.file_list = list(base_colored_images_descriptions.keys())
        self.text_list = list(base_colored_images_descriptions.values())

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.load_filepaths:
            return self.file_list[idx], self.text_list[idx]

        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if random.choice([True, False]):
            img = ImageOps.mirror(img)

        if self.transform:
            img = self.transform(img)

        text = self.text_list[idx]

        text_tokenized = convert_to_tokens_list(text)[0][0]
        if self.args.vl_negs:
            info_dict = {}
            negatives = self.negs.create_negs(text)
            info_dict.update({"negatives": negatives})
            return img, text_tokenized, info_dict

        return img, text_tokenized



class IndexRandomSampler(Sampler):
    def __init__(self, dataset, use_real=False):
        # Assumes the first dataset in the concat is the action dataset
        self.use_real = use_real
        if use_real:
            new_indices = []
            for i in np.arange(0, len(dataset), 2):
                new_indices.append([i, i+1])
        else:
            new_indices = np.arange(len(dataset))

        self.indices = new_indices

    def __iter__(self):
        for i in torch.randperm(len(self.indices)):
            if isinstance(self.indices[i], list):
                yield from iter(self.indices[i])
            else:
                yield self.indices[i]

    def __len__(self):
        return 2 * len(self.indices) if self.use_real else len(self.indices)

class descriptors:
    BIG = [
        "large",
        "big",
        "tall",
        "long",
        "huge",
        "full",
        "heavy",
        "high",
        "wide",
        "thick",
        "huge",
        "giant",
        "skinny",
        "full",
        "fat",
        "bigger",
        "taller",
        "longer",
        "heavier",
        "higher",
        "wider",
        "thicker",
    ]

    SMALL = [
        "small",
        "short",
        "little",
        "thin",
        "light",
        "narrow",
        "low",
        "tiny",
        "smaller",
        "shorter",
        "thinner",
        "lighter",
        "narrower",
        "lower",
    ]

    COMODIN = [
        "large tree",
        "large building",
        "large rock",
        "large window",
        "long neck",
        "large elephant",
        "long hair",
        "large ear",
        "tall building",
        "large bus",
        "large umbrella",
        "long tail",
        "large mirror",
        "tall tree",
        "large windows",
        "large clock",
        "large sign",
    ]
