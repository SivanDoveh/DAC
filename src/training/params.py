import argparse
import os

parent_dir = os.path.abspath(os.path.join(__file__, "..", "..",".."))

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_code_dir",
        type=str,
        default=f'{parent_dir}',
        help="root code dir",
    )

    parser.add_argument(
        "--train-data",
        type=str,
        default=f'{parent_dir}/CC3M_data/train_with_cap.csv',
        help="Path to csv file with training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=f'{parent_dir}/CC3M_data/val_with_cap.csv',
        help="Path to csv file with validation data",
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "auto"],
        default="auto",
        help="Which type of dataset to process.",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection.",
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use.",
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="file",
        help="For csv-like datasets, the name of the key for the image paths.",
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="caption",
        help="For csv-like datasets, the name of the key for the captions.",
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=5e-06, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1.0e-6, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight decay.")
    parser.add_argument(
        "--eval_merged_vl_checklist",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=5, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=1, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="How often to run evaluation with val data.",
    )
    parser.add_argument(
        "--resume",
        default="auto",
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--resume1",
        default="auto",
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--resume2",
        default="auto",
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bfloat16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action="store_true",
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action="store_true",
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action="store_true",
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--image-mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override default image mean value of dataset",
    )
    parser.add_argument(
        "--image-std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override default image std deviation of of dataset",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action="store_true",
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="torch.jit.trace the model for inference / eval only",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--debug_ip",
        default=None,
        type=str,
        help="Debug IP",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )

    parser.add_argument(
        "--debug_port",
        default=12345,
        help="Debug Port",
    )
    parser.add_argument(
        "--report-to", default="", type=str, help="Options are ['tensorboard']"
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there.",
    )

    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--freeze_text",
        action="store_true",
        help="Freeze Text Encoder, except projection layer",
    )
    parser.add_argument(
        "--freeze_visual",
        action="store_true",
        help="Freeze Visual Encoder",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument(
        "--norm_gradient_clip", type=float, default=None, help="Gradient clip."
    )

    # eval options

    parser.add_argument(
        "--vl_checklist_accuracy_jsons_folder",
        type=str,
        default="vl_checklist_accuracy_jsons_folder",
        help="vl_checklist_accuracy_jsons_folder",
    )
    parser.add_argument(
        "--vl_checklist_images_root_folder",
        type=str,
        default="vl_checklist_images_root_folder",
        help="vl_checklist_images_root_folder",
    )
    parser.add_argument(
        "--vl_checklist_jsons_with_phrases_to_images_root_folder",
        type=str,
        default="vl_checklist_annot_data",
        help="vl_checklist_jsons_with_phrases_to_images_root_folder",
    )
    parser.add_argument(
        "--vl_eval_config",
        type=str,
        default="vl_checklist/configs/clip_all.yaml",
        help="Path to config vl checklist",
    )
    parser.add_argument(
        "--eval_recall",
        action="store_true",
        help="standard clip evaluation",
    )
    parser.add_argument(
        "--eval_vl_cklist", default=False, action="store_true", help="all"
    )
    parser.add_argument(
        "--vl_neg_type",
        default=["color", "action", "material", "size", "state"],
        nargs="+",
    )
    parser.add_argument(
        "--eval_vl_cklist_all",
        default=False,
        action="store_true",
        help="eval_vl_cklist_all",
    )
    parser.add_argument("--warmup_ep", default=0, type=int, help="warmup ep")
    parser.add_argument(
        "--warmup_ep_no_bn_update", default=0, type=int, help="warmup_ep_no_bn_update"
    )
    parser.add_argument("--ZS_freq", default=0, type=int, help="ZS freq")

    # eval configs
    parser.add_argument("--ZS_steps_eval", default=False, action="store_true")
    parser.add_argument("--eval_only", default=False, action="store_true")
    parser.add_argument(
        "--no_first_eval", default=False, action="store_true", help="eval on 0"
    )
    parser.add_argument("--gradcam", default=False, action="store_true", help="gradcam")
    parser.add_argument(
        "--save_eval_model", default=False, action="store_true", help="save_eval_model"
    )

    # training using text negatives and positive
    parser.add_argument(
        "--vl_negs", default=False, action="store_true", help="vl negatives"
    )
    parser.add_argument(
        "--num_negs",
        default=1,
        type=int,
        help="number of negative examples",
    )
    parser.add_argument(
        "--neg_type",
        choices=["auto", "rand_both", "rand_three", "word_replacement", "winoground"],
        default="word_replacement",
        help="Which type of negatives to process if vl_negs training is on.",
    )
    parser.add_argument(
        "--no_neg_in_contrastive",
        default=False,
        action="store_true",
        help="no_neg_in_contrastive",
    )
    parser.add_argument(
        "--auto_neg_types", default=["VERB", "NOUN", "ADP", "ADJ", "PROPN"], nargs="+"
    )
    parser.add_argument(
        "--vl_pos",
        default=False,
        action="store_true",
        help="same semantic sentences in positive loss",
    )
    parser.add_argument(
        "--save_data",
        default=False,
        action="store_true",
        help="only do loop for saving data that created",
    )

    # for data that doesnt need to be parralel- multi gpus with multi chunks on different machines that saves output
    parser.add_argument("--chunks", default=1, type=int)
    parser.add_argument("--curr_chunk", default=0, type=int)

    # fine tune using lora
    parser.add_argument(
        "--lora",
        type=int,
        default=-1,
        help="LORA r value, default 0 means do not use LORA",
    )

    # ablations
    parser.add_argument("--kl_pos", default=False, action="store_true", help="kl_pos")
    parser.add_argument("--neg_w", type=int, default=1, help="negative loss weighting")
    parser.add_argument(
        "--freeze_img", default=False, action="store_true", help="freeze_img model"
    )
    parser.add_argument(
        "--save_bloom_neg",
        default=False,
        action="store_true",
        help="save_bloom_neg rebuttal",
    )
    parser.add_argument(
        "--save_bloom_pos", default=False, action="store_true", help="save_bloom_pos"
    )

    parser.add_argument(
        "--symmetric",
        default=False,
        action="store_true",
        help="symmetric in positives",
    )

    # radar plotting
    parser.add_argument("--radar", default=False, action="store_true", help="radar")
    parser.add_argument("--eval_radar", default=None, nargs="+")
    parser.add_argument(
        "--radar_name",
        type=str,
        default="tmp",
        help="radar_name",
    )
    parser.add_argument("--eval_radar_ep", type=str, default="5", help="eval_radar_ep")
    parser.add_argument("--start_radar", type=int, default=0, help="start_radar")
    parser.add_argument("--radar_legends", default=None, nargs="+")
    parser.add_argument("--coll", default=False, action="store_true", help="radar")

    parser.add_argument(
        "--create_quality_captions",
        default=False,
        action="store_true",
        help="create_quality_captions",
    )

    parser.add_argument(
        "--quality_captions_folder",
        type=str,
        default=f'{parent_dir}/quality_captions/',
        help="folder of QC",
    )

    parser.add_argument(
        "--create_SAM",
        default=False,
        action="store_true",
        help="create_SAM",
    )
    parser.add_argument(
        "--SAM_dense_folder",
        type=str,
        default=f'{parent_dir}/SAM_dense',
        help="folder of dense SAM",
    )

    # from caption anything
    parser.add_argument("--captioner", type=str, default="blip2")
    parser.add_argument("--segmenter", type=str, default="huge")
    parser.add_argument("--text_refiner", type=str, default="base")
    parser.add_argument(
        "--segmenter_checkpoint", type=str, default="segmenter/sam_vit_h_4b8939.pth"
    )
    parser.add_argument(
        "--seg_crop_mode",
        type=str,
        default="wo_bg",
        choices=["wo_bg", "w_bg"],
        help="whether to add or remove background of the image when captioning",
    )
    parser.add_argument(
        "--clip_filter", action="store_true", help="use clip to filter bad captions"
    )
    parser.add_argument(
        "--context_captions",
        action="store_true",
        help="use surrounding captions to enhance current caption (TODO)",
    )
    parser.add_argument(
        "--disable_regular_box",
        action="store_true",
        default=False,
        help="crop image with a regular box",
    )
    parser.add_argument("--enable_reduce_tokens", action="store_true", default=False)
    parser.add_argument("--disable_reuse_features", action="store_true", default=False)
    parser.add_argument("--enable_morphologyex", action="store_true", default=False)
    parser.add_argument(
        "--model_SAM",
        default="path to checkpoint: sam_vit_h_4b8939.pth",
        type=str,
    )

    parser.add_argument(
        "--blip_cap",
        default=False,
        action="store_true",
        help="use blip captions as positive",
    )
    parser.add_argument(
        "--common_batch_pos",
        default=False,
        action="store_true",
        help="use common_batch_pos",
    )
    parser.add_argument(
        "--calc_pos_sim", default=False, action="store_true", help="only_blip_cap"
    )
    parser.add_argument(
        "--use_only_quality_captions", default=False, action="store_true", help="only_blip_cap"
    )

    parser.add_argument(
        "--use_extra_cc3m_expanders",
        default=False,
        action="store_true",
        help="use_extra_cc3m_expanders",
    )
    parser.add_argument(
        "--use_extra_blip_cap_expanders",
        default=False,
        action="store_true",
        help="use_extra_blip_cap_expanders",
    )
    parser.add_argument(
        "--use_pre_calc_matching",
        default=False,
        action="store_true",
        help="use_pre_calc_matching",
    )
    parser.add_argument(
        "--mil_co_loader", default=False, action="store_true", help="mil_co_loader"
    )

    parser.add_argument(
        "--noun_pos", default=False, action="store_true", help="noun_pos"
    )
    parser.add_argument(
        "--path_extra_cc3m_expanders",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--path_extra_blip_cap_expanders",
        type=str,
        default=" ",
    )

    parser.add_argument(
        "--path_extra_cc3m_expanders_sen",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--path_extra_blip_cap_expanders_sen",
        type=str,
        default=" ",
    )

    parser.add_argument(
        "--path_extra_cc3m_expanders_itm_scores_sentences",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--path_extra_blip_cap_expanders_itm_scores_sentences",
        type=str,
        default=" ",
    )

    parser.add_argument(
        "--kqv_lora",
        default=False,
        action="store_true",
        help="applying lora only for k q v proj matrices",
    )

    parser.add_argument(
        "--p2t", default=False, action="store_true", help="positive to text loss"
    )

    parser.add_argument(
        "--random_sentence",
        default=False,
        action="store_true",
        help="positive to text loss",
    )
    parser.add_argument(
        "--aro_eval", default=False, action="store_true", help="positive to text loss"
    )
    parser.add_argument(
        "--avg_pos_features",
        default=False,
        action="store_true",
        help="avg_pos_features",
    )
    parser.add_argument(
        "--use_expanders_as_additional_data",
        default=False,
        action="store_true",
        help="use_expanders_as_additional_data",
    )

    # v2 desc gpt
    # cc3m
    parser.add_argument(
        "--v2_path_extra_cc3m_expanders",
        type=str,
        default=" ",
    )
    parser.add_argument("--v2_path_extra_cc3m_expanders_sen", type=str, default="")
    parser.add_argument(
        "--use_v2_extra_cc3m_expanders",
        default=False,
        action="store_true",
        help="use_extra_blip_cap_expanders",
    )

    # blip
    parser.add_argument(
        "--v2_path_extra_blip_cap_expanders",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--v2_path_extra_blip_cap_expanders_sen",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--v2_path_extra_blip_expanders_noun",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--v2_path_extra_blip_expanders_adj",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--use_v2_extra_blip_expanders",
        default=False,
        action="store_true",
        help="use_extra_blip_cap_expanders",
    )
    parser.add_argument(
        "--use_v2_extra_blip_expanders_noun",
        default=False,
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--use_v2_extra_blip_expanders_adj", default=False, action="store_true", help=""
    )

    # MIL using GPT generated sentences based on original caption
    parser.add_argument(
        "--mil_dense",
        type=str,
        default="",
        help="folder with GPT generated sentences based on original caption",
    )
    parser.add_argument(
        "--mil_dense_negs", default=False, action="store_true", help="mil_dense_negs"
    )
    parser.add_argument("--mil_batch", type=int, default=5, help="mil_batch_size")
    parser.add_argument(
        "--fast_run", default=False, action="store_true", help="Fast run to check flow"
    )
    parser.add_argument(
        "--blip_image_text_filtering",
        default=False,
        action="store_true",
        help="filter gpt sentences based on their similarity to the image base on BLIP",
    )
    parser.add_argument(
        "--mil_co_loader_type",
        type=str,
        default="max",  #'avg'
        help="mil_co_loader_type",
    )
    parser.add_argument(
        "--mil_co_loader_batch", type=int, default=5, help="mil_batch_size"
    )
    args = parser.parse_args()
    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
