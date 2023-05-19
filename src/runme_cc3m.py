import tempfile

from cvar_pyutils.ccc import submit_dependant_jobs
from cvar_pyutils.ccc import dict2params_cl
from cvar_pyutils.debugging_tools import copy_project
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--print_only",default=False,action="store_true",help="Path to csv filewith training data")
parser.add_argument("--chunks", default=1, type=int, help="chunks")
parser.add_argument('--n', type=str,default='test_db')
parser.add_argument('--debug', default=False, action="store_true")
parser.add_argument("--debug_ip", default=None, type=str, help="Debug IP")
parser.add_argument("--debug_port", default=12345, type=str, help="Debug Port")
args = parser.parse_args()

print_only = args.print_only

copy_code_to_tmp = True
tmp_root = tempfile.mkdtemp(dir='/dccstor/sivandov1/dev/open_clip_vl/tmp')  # TODO: Change the directory to a temp directory under your folder, make sure the directory exists
orig_code_root = os.path.dirname(os.path.abspath(__file__))+ "/"

if not print_only and copy_code_to_tmp:
    copy_project(orig_code_root, tmp_root,use_ext=('*.py', '*.yml','*.yaml','*.json'))
    os.chdir(tmp_root)

base_job_params={
    'number_of_rolling_jobs': 1,
    'num_nodes': 6,#6
    'num_gpus': 1,
    'num_cores': 32,
    'mem': '256g',#256
    'duration': '12h',
    'gpu_type': 'v100',# && hname!=cccxc516 && hname!=cccxc533 && hname!=cccxc507 && hname!=cccxc518  && hname!=cccxc510 && hname!=cccxc514',
}
# base_job_params={
#     'number_of_rolling_jobs': 1,
#     'num_nodes': 1,
#     'num_gpus': 1,
#     'num_cores': 1,
#     'mem': '128g',
#     'duration': '24h',
#     'gpu_type': 'a100',
# }
base_command = 'pyutils-run -m training.main'


base_params={
        'train-data': '/dccstor/aarbelle1/data/ConceptualCaptions/CC3M/train_with_cap.csv',
    # 'val-data': "/dccstor/aarbelle1/data/ConceptualCaptions/CC3M/val_with_cap.csv",
    # "eval_recall": None,
    'report-to tensorboard': None,
    'save-frequency':1,
    'csv-img-key': 'file',
    'csv-caption-key':'caption',
    'warmup': 10000,
    'batch-size': 128,
    'lr':5.0e-4,
    'wd': 0.1,
    'epochs': 10,
    'model': 'ViT-B/32',
    'logs': f'{orig_code_root}/../Outputs',
    'workers': 32,
    "beta1": 0.9,
    "beta2": 0.98,
    "eps": 1.0e-6,
    'chunks': args.chunks,
    'curr_chunk': 0,
    # "eval_vl_cklist":None,
    "save-most-recent":None,
    "zeroshot-frequency": 10,
}
experiments = [ # A dict or a tuple of two dicts, the first to expand the base_params dict for run parameters and the second to
                # expand the base_job_params dict for job submision parameters

    # {'name': 'use_v2_extra_blip_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_v2_extra_blip_expanders': None, 'use_pre_calc_matching': None,
    #  'random_sentence': None},
    # #
    # {'name': 'both_neg_use_v2_extra_blip_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_v2_extra_blip_expanders': None, 'use_pre_calc_matching': None,
    #  'random_sentence': None,"vl_negs": None, 'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},
    #
    # {'name': 'use_v2_extra_blip_expanders_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_v2_extra_blip_expanders': None, 'use_pre_calc_matching': None,
    #  'random_sentence': None,'use_expanders_as_additional_data':None},
    #
    # {'name': 'both_negs_use_v2_extra_blip_expanders_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_v2_extra_blip_expanders': None, 'use_pre_calc_matching': None,
    #  'random_sentence': None, 'use_expanders_as_additional_data': None,"vl_negs": None, 'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},

    # {'name': 'pretrained_rb_neg_blip_cap_max6_use_v2_extra_blip_expanders_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,'mil_co_loader_type':'max',
    #  'pretrained': 'openai','use_expanders_as_additional_data':None,'use_v2_extra_blip_expanders':None,'mil_co_loader':None,
    #  'resume':'/dccstor/sivandov1/dev/open_clip_vl/Outputs/rand_both_neg_only_blip_cap_2/checkpoints/epoch_5.pt','epochs': 10,
    #  },

    # {'name': 'pretrained_blip_cap_max6_use_v2_extra_blip_expanders_additional_data', 'lora': 4, 'lr': 0.0000050,
    #  'only_blip_cap_2': None, 'mil_co_loader_type': 'max',
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, 'use_v2_extra_blip_expanders': None,
    #  'mil_co_loader': None,
    #  'resume': '/dccstor/sivandov1/dev/open_clip_vl/Outputs/only_blip_cap_2/checkpoints/epoch_5.pt',
    #  'epochs': 10,},

    # {'name': 'avg6_use_v2_extra_blip_expanders_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'mil_co_loader_type': 'avg',
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, 'use_v2_extra_blip_expanders': None,
    #  'mil_co_loader': None},

    # {'name': 'mil_gpt_v2_sen_256m', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,'batch-size': 128,
    #  'pretrained': 'openai', 'mil_gpt':'/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen/'},

    # {'name': 'mil_gpt_v2_sen_noun', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen_noun/'},
    # #
    # {'name': 'mil_gpt_v2_sen_adj', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen_adj/'},

    # {'name': 'neg_mil_gpt_v2_sen', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'batch-size': 128,"vl_negs": None, 'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen/'},

    #epoch = 4
    # {'name': 'eye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'batch-size': 128,'mil_gpt_negs':None,
    #  "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],'mil_batch':2,
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen/'},

    #epoch=3
    # {'name': 'n_eye_neg_b_32_mb_5_vl_and_neg_mil_gpt_v2_sen', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'batch-size': 32, 'mil_gpt_negs': None,
    #  "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen/'},

    #epoch = 1
    # {'name': 'weye_neg_b_32_mb_5_vl_and_neg_mil_gpt_v2_sen', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'batch-size': 32, 'mil_gpt_negs': None,
    #  "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen/'},

    # {'name': 'cap_any_2mb1012', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'batch-size': 32, 'mil_gpt_negs': None,
    #  "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'mil_batch': 10,
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/cap_anything_folder/'},

    # {'name': 'my_cap_any_mb2012', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,#sivangl
    #  'batch-size': 32, 'mil_gpt_negs': None,
    #  "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'mil_batch': 20,
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/cap_anything_folder/'},
    #
    # {'name': 'my_v2_12mb20', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,'mil_batch':20,
    #  'batch-size': 32, 'mil_gpt_negs': None,
    #  "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen/'},

    # {'name': 'alpaca_64mb6', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,'mil_batch':6,
    #    'batch-size': 64, 'mil_gpt_negs': None,
    #    "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],
    #    'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/alpaca_expanders_sen/'},
    #
    {'name': 'alpaca_32mb5', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'mil_batch': 5,
     'batch-size': 32, 'mil_gpt_negs': None,
     "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],
     'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/alpaca_expanders_sen/'},

    # {'name': 'cap_any_mb10_no_milneg', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'batch-size': 32, #'mil_gpt_negs': None,
    #  "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'mil_batch': 10,
    #  'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/cap_anything_folder/'},

    {'name': 'cap_any_mb10_filtterd', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
     'batch-size': 32, 'mil_gpt_negs': None,
     "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'mil_batch': 10,
     'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/cap_anything_folder/'},

    # {'name': 'create_alpaca', 'lora': 4, 'lr': 0.0000050,'batch-size': 1, 'save_data':None,'no_first_eval': None, 'pretrained': 'openai','workers': 0, 'alpaca_expanders':'/dccstor/sivandov1/data/alpaca_cc3m_new/'},


    #create nouns splitting
    # {'name': 'NOUN_save_data_use_v2_extra_blip_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_v2_extra_blip_expanders': None, 'save_data': None,
    #  'use_v2_extra_blip_expanders_noun': None, },

    # {'name': 'adj_save_data_use_v2_extra_blip_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_v2_extra_blip_expanders': None, 'save_data': None,
    #  'use_v2_extra_blip_expanders_adj': None, },

    # {'name': 'cc3m_baseline', 'pretrained': 'openai'},#5/10
    #
    # {'name': 'cc3m_lora_2', 'lora': 4, 'lr': 0.0000050, 'pretrained': 'openai'},#5/10
    # {'name': 'blip_cap_cc3m_lora_1h', 'lora': 4, 'lr': 0.0000050,'blip_cap':None, 'pretrained': 'openai','symmetric':None,'common_batch_pos':None,'kl_pos':None},  # 5/10
    # {'name': 'v1_blip_cap_cc3m_lora_1h', 'lora': 4, 'lr': 0.0000050, 'blip_cap': None, 'pretrained': 'openai',
    #  'symmetric': None, 'common_batch_pos': None,},  # 5/10
    # {'name': 'v2_blip_cap_cc3m_lora_1h', 'lora': 4, 'lr': 0.0000050, 'blip_cap': None, 'pretrained': 'openai',
    #  'symmetric': None,'kl_pos': None},  # 5/10
    # {'name': 'v3_blip_cap_cc3m_lora_1h', 'lora': 4, 'lr': 0.0000050, 'blip_cap': None, 'pretrained': 'openai',
    #  'common_batch_pos': None, 'kl_pos': None},  # 5/10
    # {'name': 'v4_blip_cap_cc3m_lora_1h', 'lora': 4, 'lr': 0.0000050, 'blip_cap': None, 'pretrained': 'openai',},
    # {'name': 'v5_blip_cap_cc3m_lora_1h', 'lora': 4, 'lr': 0.0000050,'blip_cap':None, 'pretrained': 'openai','kl_pos':None},  # 5/10
    #
    # {'name': 'v6_blip_cap_cc3m_lora_1h', 'lora': 4, 'lr': 0.0000050,'blip_cap':None, 'pretrained': 'openai','symmetric':None,},  # 5/10
    #
    # {'name': 'v7_blip_cap_cc3m_lora_1h', 'lora': 4, 'lr': 0.0000050,'blip_cap':None, 'pretrained': 'openai','common_batch_pos':None,},  # 5/10
    # {'name': 'v8_only_blip_cap_1h', 'lora': 4, 'lr': 0.0000050, 'blip_cap': None, 'pretrained': 'openai',
    #  'only_blip_cap': None, },

    # {'name': 'only_blip_cap_2', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'pretrained': 'openai',},


    # {'name': 'only_blip_cap_1', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_1': None, 'pretrained': 'openai',},
    # {'name': 'rand_both_neg_only_blip_cap_1', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_1': None, 'pretrained': 'openai',"vl_negs": None, 'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},

    # {'name': 'rand_both_neg_only_blip_cap_2', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'pretrained': 'openai',"vl_negs": None, 'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},
    # {'name': 'RB_neg_only_blip_cap_2', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'pretrained': 'openai',"vl_negs": None,
    #  'neg_type': 'word_replacement'},
    # {'name': 'lr1_RB_neg_only_blip_cap_2', 'lora': 4, 'lr': 0.00000050, 'only_blip_cap_2': None, 'pretrained': 'openai',"vl_negs": None,
    #  'neg_type': 'word_replacement'},
    # {'name': 'lr2_RB_neg_only_blip_cap_2', 'lora': 4, 'lr': 0.000000050, 'only_blip_cap_2': None, 'pretrained': 'openai',
    #  "vl_negs": None,
    #  'neg_type': 'word_replacement'},
    # {'name': 'lr3_RB_neg_only_blip_cap_2', 'lora': 4, 'lr': 0.00050, 'only_blip_cap_2': None, 'pretrained': 'openai',
    #  "vl_negs": None,
    #  'neg_type': 'word_replacement'},

    # {'name': 'calc', 'lora': 4, 'lr': 0.0000050, 'blip_cap': None,'calc_pos_sim':None, 'pretrained': 'openai'},

    # nouns positives
    # {'name': 'nouns_pos_only_blip_cap_2', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'pretrained': 'openai',
    #  'noun_pos': None},# CHECK IF RUNS
    # {'name': 'nouns_pos_cc3m', 'lora': 4, 'lr': 0.0000050, 'pretrained': 'openai',
    #  'noun_pos': None},
    # {'name': 'common_batch_pos_nouns_pos_RB_neg_only_blip_cap_2', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', "vl_negs": None, 'noun_pos': None,'common_batch_pos':None,
    #  'neg_type': 'word_replacement'},

    # use extra blip cap expanders
    # {'name': 'use_extra_blip_cap_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None,},
    # {'name': 'neg_both_use_extra_blip_cap_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None,"vl_negs": None, 'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},

    # {'name': 'avg_pos_features', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,'avg_pos_features':None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None,"vl_negs": None, 'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},

    # {'name': 'symmetric_avg_pos_features', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'avg_pos_features': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None, "vl_negs": None, 'neg_type': 'rand_both','symmetric':None,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},
    #
    # {'name': 'common_batch_pos_avg_pos_features', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'avg_pos_features': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None, "vl_negs": None, 'neg_type': 'rand_both','common_batch_pos':None,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},
    #
    # {'name': 'kl_pos_avg_pos_features', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'avg_pos_features': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None, "vl_negs": None, 'neg_type': 'rand_both','kl_pos':None,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},

    # {'name': 'symmetric_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None, "vl_negs": None, 'neg_type': 'rand_both','symmetric':None,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},
    #
    # {'name': 'common_batch_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None, "vl_negs": None, 'neg_type': 'rand_both','common_batch_pos':None,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},
    #
    # {'name': 'kl_pos_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None, "vl_negs": None, 'neg_type': 'rand_both','kl_pos':None,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},

    # co loader
    # {'name': 'RB_use_expanders_as_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, "vl_negs": None, 'neg_type': 'word_replacement',},
    #
    # {'name': 'color_RB_use_expanders_as_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, "vl_negs": None,
    #  'neg_type': 'word_replacement','vl_neg_type':['color'] },
    #
    # {'name': 'action_RB_use_expanders_as_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, "vl_negs": None,
    #  'neg_type': 'word_replacement', 'vl_neg_type': ['action']},
    #
    # {'name': 'material_RB_use_expanders_as_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, "vl_negs": None,
    #  'neg_type': 'word_replacement', 'vl_neg_type': ['material']},
    #
    # {'name': 'state_RB_use_expanders_as_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, "vl_negs": None,
    #  'neg_type': 'word_replacement', 'vl_neg_type': ['state']},

    # {'name': 'size_RB_use_expanders_as_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, "vl_negs": None,
    #  'neg_type': 'word_replacement', 'vl_neg_type': ['size']},

    # {'name': 'save_data_use_v2_extra_blip_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_v2_extra_blip_expanders': None,'save_data':None},


    # {'name': 'use_expanders_as_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None},
    #
    # {'name': 'both_negs_use_expanders_as_additional_data', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_expanders_as_additional_data': None, "vl_negs": None, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},

    #uniform
    # {'name': 'ones_neg_both_use_extra_blip_cap_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None, "vl_negs": None, 'neg_type': 'rand_both','do_not_use_blip_match':None,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},

    # {'name': 'use_extra_cc3m_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_cc3m_expanders': None,},
    # {'name': 'neg_both_use_extra_cc3m_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_cc3m_expanders': None,"vl_negs": None, 'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB']},
    #
    # kqv lora
    # {'name': 'kqv_lora_RB_neg_blip_cap_2', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None, 'pretrained': 'openai',"vl_negs": None,
    #  'neg_type': 'word_replacement','kqv_lora':None},


    # create blip_cap_expanders
    # {'name': 'save_data_use_extra_blip_cap_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None,'save_data':None},
    # {'name': 'save_data_use_extra_blip_cap_expanders', 'lora': 4, 'lr': 0.0000050, 'only_blip_cap_2': None,
    #  'pretrained': 'openai', 'use_extra_blip_cap_expanders': None,},

    # create cap anything captions
    # {'name': 'create_cap_anything_1','no_first_eval': None, 'create_cap_anything': None,'save_data':None, 'batch-size': 1, 'pretrained': 'openai','workers': 0,},

    # 5/10
    #

    # {'name': 'cc3m_rb_neg',
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'word_replacement','pretrained': 'openai'},#5/10
    # {'name': 'cc3m_pos',
    #  'lora': 4, 'lr': 0.0000050, "vl_pos": None,'pretrained': 'openai'},"p2t":None

    #
    # {'name': 'cc3m_rb_neg',
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'word_replacement','pretrained': 'openai'},#5/10

    # {'name': 'cc3m_auto_neg',#5/10
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'auto',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],'pretrained': 'openai',},
    #
    # {'name': 'cor_cc3m_both_negs', #5,10
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],'pretrained': 'openai'},
    #
    # {'name': 'cor_cc3m_pos_both_negs',#5,10
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], "vl_pos": None, 'pretrained': 'openai'},"p2t":None
#ablations
    # {'name': 'vl_bloom_neg_1',
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'vl_bloom_neg':None, 'pretrained': 'openai'},

    # {'name': 'cc3m_pos_symmetric_6h',
    #  'lora': 4, 'lr': 0.0000050, "vl_pos": None, "symmetric": None, 'pretrained': 'openai'},
# qm
    # {'name': 'cc3m_pos_qm_v3',
    #  'lora': 4, 'lr': 0.0000050, "vl_pos": None, 'pretrained': 'openai'},#5/10
    # {'name': 'cc3m_pos_both_negs_qm_v3',#5,10
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], "vl_pos": None, 'pretrained': 'openai'},
#forgeting analysis
    # {'name': 'f100_base_both_negs_ZS', 'ZS_steps_eval': None, 'epochs': 3,'ZS_freq':100,
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'pretrained': 'openai'},

    # {'name': 'f25_base_both_negs_ZS', 'ZS_steps_eval': None, 'epochs': 3, 'ZS_freq': 25,
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'pretrained': 'openai'},
    #
    # {'name': 'f10_base_both_negs_ZS', 'ZS_steps_eval': None, 'epochs': 3, 'ZS_freq': 10,
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'pretrained': 'openai'},

    # {'name': 'f1_base_ZS', 'ZS_steps_eval': None, 'epochs': 3, 'ZS_freq': 1, 'pretrained': 'openai'},
    # {'name': 'f10_base_ZS', 'ZS_steps_eval': None, 'epochs': 3, 'ZS_freq': 10,'pretrained': 'openai'},
    # {'name': 'f100_base_ZS', 'ZS_steps_eval': None, 'epochs': 3, 'ZS_freq': 100, 'pretrained': 'openai'},
    # {'name': 'f50_base_ZS', 'ZS_steps_eval': None, 'epochs': 3, 'ZS_freq': 50, 'pretrained': 'openai'},
    # {'name': 'f25_base_ZS', 'ZS_steps_eval': None, 'epochs': 3, 'ZS_freq': 25, 'pretrained': 'openai'},
    # {'name': 'f100_warmup_1_both_negs_ZS', 'ZS_steps_eval':None,'epochs': 3,'ZS_freq':100,
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both','warmup_ep':1,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],'pretrained': 'openai'},
    #
    # {'name': 'f100_warmup_2_both_negs_ZS', 'ZS_steps_eval': None, 'epochs': 3,'ZS_freq':100,
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both', 'warmup_ep': 2,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'pretrained': 'openai'},
    #
    # {'name': 'f100_no_bn_update_1_both_negs_ZS', 'ZS_steps_eval': None, 'epochs': 3,'ZS_freq':100,
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both', 'warmup_ep_no_bn_update': 1,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'pretrained': 'openai'},



## lorot
    # {'name': 's_ora_2_lr_8_cc3m_pos_both_negs',
    #  "vl_negs": None, 'lora': 2, 'lr': 0.000000050, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], "vl_pos": None, 'pretrained': 'openai'},

    # {'name': 'lora_4_lr_10_cc3m_pos_both_negs',
    #  "vl_negs": None, 'lora': 4, 'lr': 0.00000000050, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], "vl_pos": None, 'pretrained': 'openai'},

# different neg exp
    # {'name': 'neg_w_0_no_seperate_neg_loss_pos',
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both','neg_w':0,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], "vl_pos": None, 'pretrained': 'openai'},

    # {'name': 'no_neg_in_contrastive',
    #  "vl_negs": None, 'lora': 4, 'lr': 0.0000050, 'neg_type': 'rand_both','no_neg_in_contrastive':None,
    #  "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],'pretrained': 'openai'},

###########********** from scratch ************##########Leo

    # {'name': 'Leo_fs_cc3m_baseline', 'lr': 0.0000050,},#5,10
    # {'name': 'fs_cc3m_pos', "vl_pos": None,},#5/10
    # {'name': 'fs_cc3m_rb_neg',"vl_negs": None, 'neg_type': 'word_replacement',},#5/10
    # {'name': 'L_fs_cc3m_both_negs',"vl_negs": None,'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],},#5
    # {'name': 'fs_cc3m_both_negs',"vl_negs": None,'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],},#5

    #
    # {'name': 'fs_cc3m_pos_both_negs', "vl_negs": None,'neg_type': 'rand_both',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], "vl_pos": None,},#5,10
    # {'name': 'fs_cc3m_pos_both_negs_no_adp', "vl_negs": None, 'neg_type': 'rand_both',
    #  "auto_neg_types": ['NOUN','ADJ', 'VERB'], "vl_pos": None, },  # 5,10

    # {'name': 'L_fs_cc3m_auto_neg',"vl_negs": None,'neg_type': 'auto',"auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],},#5,10
]

for ind, experiment in enumerate(experiments):
    params_, job_params_ = experiment if isinstance(experiment,tuple) else (experiment, {})
    params = base_params.copy()
    job_params = base_job_params.copy()
    params.update(params_)
    job_params.update(job_params_)

    if job_params.pop('skip', False):
        continue
    if 'name' not in params.keys():
        print(f'Could not run experiment number {ind}. Missing name for experiment')
        continue
    job_params['name'] = params['name']
    if print_only:
        params['name'] = args.n
        params['workers'] = 0
        params['debug'] = None
        params['no_first_eval'] = None
    if args.chunks>1:
        params.pop('name')
    for curr_chunk in range(args.chunks):
        if args.chunks > 1:
            job_params['name'] = f'curr_chunk:{curr_chunk}'
        else:
            job_params['name'] = f'{job_params["name"]}'
        params['curr_chunk'] = curr_chunk
        cmnd = dict2params_cl(params, base_command)
        if not print_only:
            all_job_ids, all_job_outputs = submit_dependant_jobs(command_to_run=cmnd, **job_params)
            for jid, jout in zip(all_job_ids, all_job_outputs):
                print(f'Submitted job: {jid}:\n\tCommand:{cmnd}\n\tOutput: {jout}')
        else:
            print(f'Job number: {ind}\n{cmnd}')




