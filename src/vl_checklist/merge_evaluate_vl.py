from vl_checklist.utils import chunks
from vl_checklist.data_loader import DataLoader
from tqdm import tqdm
import yaml
import os
import random
import time
import json
import open_clip as clip
from PIL import Image
import torch
import logging
from torch import distributed as dist
from open_clip import create_model_and_transforms


def is_master(args):
    return (not args.distributed) or args.rank == 0


def EvaluateAllVL_merged(preprocess_val, start_epoch, args, writer):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    args.device = device
    model1, _, image_preprocess = create_model_and_transforms(
        "ViT-B/32",
        "openai",
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        lora=args.lora,
        freeze_img=args.freeze_img,
        kqv_lora=args.kqv_lora,
    )
    if args.resume1 != "None":
        checkpoint = torch.load(args.resume1, map_location="cpu")
        print("Load from path:", args.resume1)
        sd = checkpoint["state_dict"]

        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}

        model1.load_state_dict(sd)
    else:
        print("no checkpoint")
        args.resume1 = "Output_no_checkpoint/"

    model2, _, image_preprocess = create_model_and_transforms(
        "ViT-B/32",
        "openai",
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        lora=args.lora,
        freeze_img=args.freeze_img,
        kqv_lora=args.kqv_lora,
    )
    if args.resume2 != "None":
        checkpoint = torch.load(args.resume2, map_location="cpu")
        print("Load from path:", args.resume2)
        sd = checkpoint["state_dict"]

        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}

        model2.load_state_dict(sd)
    else:
        print("no checkpoint")
        args.resume2 = "Output_no_checkpoint/"
    model1.eval()
    model2.eval()
    if args.eval_vl_cklist_all:
        vl_eval = Evaluate(
            config_file="vl_checklist/configs/clip_all_obj.yaml",
            model1=model1,
            model2=model2,
            preprocess_val=preprocess_val,
            epoch=start_epoch,
            args=args,
            tb_writer=writer,
        )
        vl_eval.start()

    vl_eval = Evaluate(
        config_file="vl_checklist/configs/clip_all_attribute.yaml",
        model1=model1,
        model2=model2,
        preprocess_val=preprocess_val,
        epoch=start_epoch,
        args=args,
        tb_writer=writer,
    )
    vl_eval.start()

    vl_eval = Evaluate(
        config_file="vl_checklist/configs/clip_all_rel.yaml",
        model1=model1,
        model2=model2,
        preprocess_val=preprocess_val,
        epoch=start_epoch,
        args=args,
        tb_writer=writer,
    )
    vl_eval.start()

    vl_eval = Evaluate(
        config_file="vl_checklist/configs/clip_all_rel_spatial.yaml",
        model1=model1,
        model2=model2,
        preprocess_val=preprocess_val,
        epoch=start_epoch,
        args=args,
        tb_writer=writer,
    )
    vl_eval.start()


class Evaluate(object):
    def __init__(
        self, config_file, model1, model2, preprocess_val, epoch, args, tb_writer=None
    ) -> None:
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.log_dir = os.path.join(args.logs, args.name)
        m = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        self.batch_size = m["BATCH_SIZE"]
        self.model1 = model1.module if args.distributed else model1
        self.model2 = model2.module if args.distributed else model2
        self.alpha = args.alpha
        self.max_num = m["MAX_NUM"]
        self.data_names = m["DATA"]["TEST_DATA"]
        self.task = m["TASK"]
        self.types = m["DATA"]["TYPES"]
        self.dir = m["OUTPUT"]["DIR"]
        self.sample_num = m["OUTPUT"]["NUM"]
        self.model_name = self.model1.model_name
        self.preprocess_val_clip = preprocess_val
        self.epoch = epoch
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tb_writer = tb_writer
        self.config_file = config_file
        self.saliency_layer = "layer4"

    def start(self):
        avg = 0
        for data_type in self.types:
            results = self.eval(data_type=data_type)
            avg += results
        avg = avg / len(self.types)
        if self.args.save_logs:
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(
                    f"val/{self.types[0].split('/')[0]}_avg_eval", avg, self.epoch
                )
                logging.info(
                    f" AVG {self.epoch}: {self.types[0].split('/')[0]}_avg_eval {avg}"
                )

    def clip_model_wrapper(self, images, texts):
        probs = []
        for i, chunk_i in enumerate(chunks(images, self.batch_size)):
            for j in range(len(chunk_i)):
                image = (
                    self.preprocess_val_clip(Image.open(chunk_i[j]))
                    .unsqueeze(0)
                    .to(self.device)
                )
                text = clip.tokenize(texts[j]).to(self.device)
                with torch.no_grad():
                    image_features1, text_features1, logit_scale1 = self.model1(
                        image, text
                    )
                    image_features2, text_features2, logit_scale2 = self.model2(
                        image, text
                    )
                    logits_per_image1 = (
                        logit_scale1 * image_features1 @ text_features1.t()
                    )
                    logits_per_image2 = (
                        logit_scale2 * image_features2 @ text_features2.t()
                    )
                    logits_per_image = (
                        self.alpha * logits_per_image1
                        + (1 - self.alpha) * logits_per_image2
                    )
                    probs.extend(logits_per_image.cpu().numpy())
        return {"probs": probs}

    def eval(self, data_type):
        max_number = self.max_num
        d = DataLoader(self.data_names, self.args, data_type, self.task)
        results = {}
        index = 0

        if self.task == "itc":
            for name in d.data:
                if not is_master(self.args):
                    continue
                sample_true = []
                sample_false = []
                num_t, num_f = 0, 0
                if max_number:
                    d.data[name] = d.data[name][: int(max_number / 2)]
                starttime = time.time()
                for batch in tqdm(
                    chunks(d.data[name], self.batch_size),
                    desc="Progress",
                    ncols=100,
                    total=int(len(d.data[name]) / self.batch_size),
                ):
                    images = [z["path"] for z in batch]
                    texts_pos = [z["texts_pos"][0] for z in batch]
                    texts_neg = [z["texts_neg"][0] for z in batch]

                    result_pos = self.clip_model_wrapper(images, texts_pos)
                    result_neg = self.clip_model_wrapper(images, texts_neg)

                    result_t1 = zip(result_pos["probs"], result_neg["probs"])
                    result_tmp = list(result_t1)

                    for i in range(len(result_tmp)):
                        index = index + 1
                        if result_tmp[i][0][0] > result_tmp[i][1][0]:
                            sample_true.append(
                                {
                                    "img_path": images[i],
                                    "pos_score": float(round(result_tmp[i][0][0], 4)),
                                    "pos_txt": texts_pos[i],
                                    "neg_score": float(round(result_tmp[i][1][0], 4)),
                                    "neg_txt": texts_neg[i],
                                    "result": "correct",
                                }
                            )
                            num_t += 1

                        else:
                            sample_false.append(
                                {
                                    "img_path": images[i],
                                    "pos_score": float(round(result_tmp[i][0][0], 4)),
                                    "pos_txt": texts_pos[i],
                                    "neg_score": float(round(result_tmp[i][1][0], 4)),
                                    "neg_txt": texts_neg[i],
                                    "result": "incorrect",
                                }
                            )
                            num_f += 1

                endtime = time.time()
                accuracy = float(num_t) / (num_t + num_f)
                results[name] = round(accuracy, 4)
                file_name = data_type.replace("/", "_")
                path = os.path.join(
                    self.args.logs.split("//")[0].split("src")[0],
                    "eval_jsons_correct",
                    self.dir.split("/")[1],
                    "merged_models_" + str(self.alpha),
                )

                os.makedirs(path, exist_ok=True)
                with open(
                    os.path.join(path, f"{file_name}_{name}_{self.epoch}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "total_acc": round(accuracy, 4),
                            "number_of_data": len(d.data[name]),
                            "model_name": self.model_name,
                            "task": self.task,
                            "eval_time": endtime - starttime,
                        },
                        f,
                    )

                logging.info(
                    f"Eval {name} VL Epoch: {self.epoch} {data_type}_eval: {round(accuracy, 4)}"
                )

                if self.args.save_logs:
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar(
                            f"val/{name}/{data_type}_eval",
                            round(accuracy, 4),
                            self.epoch,
                        )

        if self.args.save_logs and is_master(self.args):
            if self.tb_writer is not None:
                both_res = 0
                for k in results.keys():
                    both_res += results[k]
                both_res = both_res / results.keys().__len__()
                self.tb_writer.add_scalar(
                    f"val/both/{data_type}_eval", both_res, self.epoch
                )

                return both_res
        return 0
