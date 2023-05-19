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

    parser.add_argument("--images_names_csv", type=str, default="/dccstor/aarbelle1/data/ConceptualCaptions/CC3M/train_with_cap.csv",
                        help="images_names_csv", )
    # parser.add_argument("--texts_expanding_folder", type=str, default="/dccstor/leonidka1/data/cc3m_LLM_outputs/GPT_NEO",
    #                     help="texts_expanding_folder", )
    parser.add_argument(
        "--cc3m_v2",
        default=False,
        action="store_true",
        help="the descriptions list of paola"
    )
    parser.add_argument(
        "--save_sentences",
        default=False,
        action="store_true",
        help="save sentences and not only save texts"
    )


    parser.add_argument("--save_sentences_path", type=str, default="/dccstor/sivandov1/data/evlk/v2_cc3m_divided_to_sentences",
                        help="save_sentences_path", )
    parser.add_argument("--save_renamed_text_expanders", type=str, default="/dccstor/sivandov1/data/evlk/v2_cc3m_GPT_NEO_text_expander",
                        help="save_text_expanders", )
    args = parser.parse_args()
    # If some params are not passed, we use the default values based on model name.
    # for name, val in default_params.items():
    #     if getattr(args, name) is None:
    #         setattr(args, name, val)

    return args
