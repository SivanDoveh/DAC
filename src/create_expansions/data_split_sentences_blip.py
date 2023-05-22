from torch.utils.data import Dataset, DataLoader, Sampler
import os
import nltk
import pandas as pd
import torch
import re
import json

# Load the pre-trained sentence tokenizer
nltk.download("punkt")
sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


class TextDataset(Dataset):
    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.args = args
        folder_path = "/dccstor/leonidka1/data/cc3m_LLM_outputs/GPT_NEO_LIST_DESC/"
        self.file_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
        ]  # if "good_mapping" in f]
        self.save_sentences_path = args.save_sentences_path
        self.labels = "/dccstor/leonidka1/data/cc3m_LLM_outputs/image_labels.tsv"
        df1 = pd.read_csv(self.labels, sep="\t")
        column_names = df1.columns.tolist()
        df1 = pd.DataFrame([column_names] + df1.values.tolist(), columns=column_names)
        df1 = df1.rename(columns={df1.iloc[0, 0]: "caption"})
        self.captions = df1.iloc[:, 0].to_list()
        self.train_with_cap = pd.read_csv(args.images_names_csv, sep="\t")
        self.captions2 = self.train_with_cap["caption"].to_list()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        example_num = self.file_paths[idx].split("/")[-1].strip(".txt").split("_")[-1]
        caption = self.captions[int(example_num)]
        try:
            ind = self.captions2.index(caption)
            image_name = self.train_with_cap["file"][ind].split("/")[-1]
        except:
            print(
                f"The file {example_num} doesn't have image or caption does not have expander"
            )
            return []
        if self.args.save_sentences:
            if os.path.exists(f"{self.save_sentences_path}/{image_name}.json"):
                print(f"The file {image_name}.json exists in the folder.")
                return []

        with open(self.file_paths[idx], "r") as f:
            curr_text = f.read()
        curr_text_clean = re.sub(" +", " ", curr_text).replace("\n", "")

        curr_text_clean = curr_text_clean.split("short:")[0]

        with open(
            f"{self.args.save_renamed_text_expanders}/{image_name}.txt", "w"
        ) as f:
            json.dump(curr_text_clean, f)
        os.chmod(f"{self.args.save_renamed_text_expanders}/{image_name}.txt", 0o0777)

        if self.args.save_sentences:
            sentences_list = sent_tokenizer.tokenize(curr_text_clean)
            if self.args.cc3m_v2:
                sentences_list = [item.split("\t")[1] for item in sentences_list[1:]]
            # Save the list of sentences to a JSON file
            with open(f"{self.args.save_sentences_path}/{image_name}.json", "w") as f:
                json.dump(sentences_list, f)
            os.chmod(f"{self.args.save_sentences_path}/{image_name}.json", 0o0777)
        print(f"{image_name}")
        print(f"caption: {caption}")
        print(f"text: {curr_text_clean}")
        print(f"sentences_list:")
        for s in sentences_list:
            print(f"{s}")
        return []


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


def get_dataloader(args):
    dataset = TextDataset(args)
    sampler = None
    if args.chunks > 1 and args.save_data:
        sampler = ChunkSample(
            dataset, shuffle=False, rank=args.curr_chunk, num_replicas=args.chunks
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
    )
    return dataloader


def bla():
    images_folder = "/dccstor/sivandov1/data/evlk/sample_images"
    images_names = os.listdir(images_folder)
    for img_name in images_names:
        print(img_name)
        f = open(
            f'{os.path.join("/dccstor/sivandov1/data/blip2_positives_cc3m/", img_name)}.json'
        )
        caption = json.load(f)
        print(f"caption: {caption['positive_caption'][0]}")
        folder_path = "/dccstor/leonidka1/data/cc3m_LLM_outputs/blip2_positives_cc3m_gpt_extra/GPT_NEO"
        expanded_caption = os.path.join(folder_path, img_name + ".txt")
        with open(expanded_caption, "r") as f:
            curr_text = f.read()
        curr_text_clean = re.sub(" +", " ", curr_text).replace("\n", "")

        # clean short
        curr_text_clean = curr_text_clean.split("short description:")[0]
        temp = curr_text_clean.split("you might see the following:")
        if len(temp) > 1:
            curr_text_clean = temp[1]
        temp = curr_text_clean.split("you might see a scene where")
        if len(temp) > 1:
            curr_text_clean = temp[1]

        sent_list = sent_tokenizer.tokenize(curr_text_clean)
        sent_list = [s.strip("*") for s in sent_list]
        print(sent_list)
