import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_code_dir",
        type=str,
        default="/dccstor/sivandov1/dev/open_clip_vl/",
        help="root code dir",
    )

    parser.add_argument(
        "--debug_ip",
        default=None,
        type=str,
        help="Debug IP",
    )

    parser.add_argument(
        "--debug_port",
        default=12345,
        help="Debug Port",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument('--chunks', default=1, type=int)
    parser.add_argument('--curr_chunk', default=0, type=int)
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--save_data",
        default=False,
        action="store_true",
        help="save data"
    )

    parser.add_argument("--synt_captions", type=str, default="/dccstor/paolac1/data/syn4vl_dataset_captions",
                        help="synt_captions", )
    parser.add_argument('--vl_neg_type', default=['color', 'action', 'material', 'size', 'state'], nargs='+')
    parser.add_argument("--vl_negs", default=True, action="store_true", help="vl negatives")
    parser.add_argument("--num_negs",default=1,type=int,help="number of negative examples",)
    parser.add_argument("--save_dir", type=str, default="/dccstor/sivandov1/data/paola_negs",
                        help="syn4vl_captions_negatives", )
    args = parser.parse_args()

    return args
