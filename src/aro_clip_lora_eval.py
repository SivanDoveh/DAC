import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.params import parse_args
from open_clip import create_model_and_transforms
from aro.dataset_zoo import VG_Relation, VG_Attribution
from aro.dataset_zoo import COCO_Order, Flickr30k_Order
import csv
from open_clip import tokenize 


def interpolate_models(theta_0, theta_1, alpha):
    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()
    }
    return theta


@torch.no_grad()
def get_retrieval_scores_batched(model, joint_loader, device="cuda"):
    """Computes the scores for each image_option / caption_option pair in the joint loader.
    Args:
        joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
        "image_options" is a list of images, and "caption_options" is a list of captions.
    Returns:
        all_scores: A numpy array containing the scores of the shape NxKxL,
        where N is the number of test cases, K is the number of image options per the test case,
        and L is the number of caption options per the test case.
    """
    scores = []
    tqdm_loader = tqdm(joint_loader)
    tqdm_loader.set_description("Computing retrieval scores")
    for batch in tqdm_loader:
        image_options = []
        for i_option in batch["image_options"]:
            # image_embeddings = model.encode_image(i_option.to(device)).cpu().numpy() # B x D
            image_embeddings = model(i_option.to(device), None).cpu().numpy()  # B x D
            image_embeddings = image_embeddings / np.linalg.norm(
                image_embeddings, axis=1, keepdims=True
            )  # B x D
            image_options.append(np.expand_dims(image_embeddings, axis=1))

        caption_options = []
        for c_option in batch["caption_options"]:
            caption_tokenized = torch.cat([tokenize(c) for c in c_option])
            caption_embeddings = (
                model(None, caption_tokenized.to(device)).cpu().numpy()
            )  # B x D

            # caption_embeddings = model.encode_text(caption_tokenized.to(device)).cpu().numpy() # B x D
            caption_embeddings = caption_embeddings / np.linalg.norm(
                caption_embeddings, axis=1, keepdims=True
            )  # B x D
            caption_options.append(np.expand_dims(caption_embeddings, axis=1))

        image_options = np.concatenate(image_options, axis=1)  # B x K x D
        caption_options = np.concatenate(caption_options, axis=1)  # B x L x D
        batch_scores = np.einsum(
            "nkd,nld->nkl", image_options, caption_options
        )  # B x K x L
        scores.append(batch_scores)

    all_scores = np.concatenate(scores, axis=0)  # N x K x L
    return all_scores


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.debug:
        if args.debug_ip is None:
            import pydevd_pycharm

            pydevd_pycharm.settrace(
                os.environ["SSH_CONNECTION"].split()[0],
                port=args.debug_port,
                stdoutToServer=True,
                stderrToServer=True,
                suspend=False,
            )

    # model, image_preprocess = clip.load("ViT-B/32", jit=False, lora=lora)
    model, _, image_preprocess = create_model_and_transforms(
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
    if args.resume != "None":
        checkpoint = torch.load(args.resume, map_location="cpu")
        print("Load from path:", args.resume)
        sd = checkpoint["state_dict"]

        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}

        model.load_state_dict(sd)
    else:
        print("no checkpoint")
        args.resume = "Output_no_checkpoint/"
    model.eval()
    root_root_dir = "aro"
    root_dir = f"{root_root_dir}/datasets"
    vgr_dataset = VG_Relation(
        image_preprocess=image_preprocess, download=True, root_dir=root_dir
    )
    vgr_loader = DataLoader(vgr_dataset, batch_size=500, shuffle=False)
    vgr_scores = get_retrieval_scores_batched(model, vgr_loader)
    # Evaluate the macro accuracy
    vgr_records = vgr_dataset.evaluate_scores(vgr_scores)
    symmetric = [
        "adjusting",
        "attached to",
        "between",
        "bigger than",
        "biting",
        "boarding",
        "brushing",
        "chewing",
        "cleaning",
        "climbing",
        "close to",
        "coming from",
        "coming out of",
        "contain",
        "crossing",
        "dragging",
        "draped over",
        "drinking",
        "drinking from",
        "driving",
        "driving down",
        "driving on",
        "eating from",
        "eating in",
        "enclosing",
        "exiting",
        "facing",
        "filled with",
        "floating in",
        "floating on",
        "flying",
        "flying above",
        "flying in",
        "flying over",
        "flying through",
        "full of",
        "going down",
        "going into",
        "going through",
        "grazing in",
        "growing in",
        "growing on",
        "guiding",
        "hanging from",
        "hanging in",
        "hanging off",
        "hanging over",
        "higher than",
        "holding onto",
        "hugging",
        "in between",
        "jumping off",
        "jumping on",
        "jumping over",
        "kept in",
        "larger than",
        "leading",
        "leaning over",
        "leaving",
        "licking",
        "longer than",
        "looking in",
        "looking into",
        "looking out",
        "looking over",
        "looking through",
        "lying next to",
        "lying on top of",
        "making",
        "mixed with",
        "mounted on",
        "moving",
        "on the back of",
        "on the edge of",
        "on the front of",
        "on the other side of",
        "opening",
        "painted on",
        "parked at",
        "parked beside",
        "parked by",
        "parked in",
        "parked in front of",
        "parked near",
        "parked next to",
        "perched on",
        "petting",
        "piled on",
        "playing",
        "playing in",
        "playing on",
        "playing with",
        "pouring",
        "reaching for",
        "reading",
        "reflected on",
        "riding on",
        "running in",
        "running on",
        "running through",
        "seen through",
        "sitting behind",
        "sitting beside",
        "sitting by",
        "sitting in front of",
        "sitting near",
        "sitting next to",
        "sitting under",
        "skiing down",
        "skiing on",
        "sleeping in",
        "sleeping on",
        "smiling at",
        "sniffing",
        "splashing",
        "sprinkled on",
        "stacked on",
        "standing against",
        "standing around",
        "standing behind",
        "standing beside",
        "standing in front of",
        "standing near",
        "standing next to",
        "staring at",
        "stuck in",
        "surrounding",
        "swimming in",
        "swinging",
        "talking to",
        "topped with",
        "touching",
        "traveling down",
        "traveling on",
        "tying",
        "typing on",
        "underneath",
        "wading in",
        "waiting for",
        "walking across",
        "walking by",
        "walking down",
        "walking next to",
        "walking through",
        "working in",
        "working on",
        "worn on",
        "wrapped around",
        "wrapped in",
        "by",
        "of",
        "near",
        "next to",
        "with",
        "beside",
        "on the side of",
        "around",
    ]
    df = pd.DataFrame(vgr_records)
    df = df[~df.Relation.isin(symmetric)]
    print(f"VG-Relation Macro Accuracy: {df.Accuracy.mean()}")
    vg_relation_macro_acc = df.Accuracy.mean()

    vga_dataset = VG_Attribution(
        image_preprocess=image_preprocess, download=True, root_dir=root_dir
    )
    vga_loader = DataLoader(vga_dataset, batch_size=100, shuffle=False)
    vga_scores = get_retrieval_scores_batched(model, vga_loader, device="cuda")
    # Evaluate the macro accuracy
    vga_records = vga_dataset.evaluate_scores(vga_scores)
    df = pd.DataFrame(vga_records)
    print(f"VG-Attribution Macro Accuracy: {df.Accuracy.mean()}")
    vg_attribution_macro_acc = df.Accuracy.mean()

    root_dir = f"{root_root_dir}/COCO"
    coco_order_dataset = COCO_Order(
        image_preprocess=image_preprocess,
        download=True,
        root_dir=root_dir,
        split="test",
    )
    coco_loader = DataLoader(coco_order_dataset, batch_size=100, shuffle=False)
    coco_scores = get_retrieval_scores_batched(model, coco_loader, device="cuda")
    coco_result_records = coco_order_dataset.evaluate_scores(coco_scores)
    for record in coco_result_records:
        print(root_dir, record)
        coco_order = list(record.values())[0]

    root_dir = f"{root_root_dir}/Flickr30k"
    flickr_order_dataset = Flickr30k_Order(
        image_preprocess=image_preprocess, split="test", root_dir=root_dir
    )
    flickr_loader = DataLoader(flickr_order_dataset, batch_size=100, shuffle=False)
    flickr_scores = get_retrieval_scores_batched(model, flickr_loader, device="cuda")
    flickr_result_records = flickr_order_dataset.evaluate_scores(flickr_scores)
    for record in flickr_result_records:
        print(root_dir, record)
        flickr30k_order = list(record.values())[0]

    res_file = f"results_frozen.csv"
    with open(res_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                args.resume,
                vg_relation_macro_acc,
                vg_attribution_macro_acc,
                coco_order,
                flickr30k_order,
            ]
        )
