import tempfile
from cvar_pyutils.ccc import submit_dependant_jobs
from cvar_pyutils.ccc import dict2params_cl
from cvar_pyutils.debugging_tools import copy_project
import os
import argparse
import seaborn as sns

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
    'num_nodes': 1,
    'num_gpus': 0,
    'num_cores': 4,
    'mem': '32g',
    'duration': '1h',
    'gpu_type': 'a100 || v100 && hname!=cccxc547',
    # 'project_name': 'VL',
}
# base_command = 'python -m cvar_pyutils.pytorch_utils.launch -m training.main'
# base_command = 'pyutils-launch -m training.main'
base_command = 'pyutils-run -m training.main'


base_params={
    'train-data': '/dccstor/aarbelle1/data/ConceptualCaptions/CC3M/train_with_cap.csv', # on laion400 there is only pos+neg
    'val-data': "/dccstor/aarbelle1/data/ConceptualCaptions/CC3M/val_with_cap.csv",
    'report-to tensorboard': None,
    'save-frequency':10,
    'csv-img-key': 'file',
    'csv-caption-key':'caption',
    'warmup': 10000,
    'batch-size': 128,
    'lr':5.0e-4,
    'wd': 0.1,
    'epochs': 15,
    'model': 'ViT-B/32',
    'logs': f'{orig_code_root}/../Outputs',
    'workers': 16,
    "beta1": 0.9,
    "beta2": 0.98,
    "eps": 1.0e-6,
    'pretrained':'openai',
    'chunks': args.chunks,
    'curr_chunk': 0,
    "radar":None,

}

experiments = [ # A dict or a tuple of two dicts, the first to expand the base_params dict for run parameters and the second to
                # expand the base_job_params dict for job submision parameters
    #
    # {'name': 'blip_cap_cc3m_lora_1h', 'radar': None, 'radar_name': 'blip_cap_cc3m_lora_1h','radar_legends':['clip','v8_only_blip_cap_1h','only_blip_cap_2','RB_neg_only_blip_cap_2','rand_both_neg_only_blip_cap_2'],'eval_radar_ep':5,
    #  'eval_radar': ['clip','v8_only_blip_cap_1h','only_blip_cap_2','RB_neg_only_blip_cap_2','rand_both_neg_only_blip_cap_2'],'start_radar':30},
    # {'name': 'radar', 'radar': None, 'radar_name': 'noun',
    #  'radar_legends': ['clip', 'use_extra_cc3m_expanders','neg_both_use_extra_cc3m_expanders'], 'eval_radar_ep': 5,
    #  'eval_radar': ['clip','use_extra_cc3m_expanders','neg_both_use_extra_cc3m_expanders'], 'start_radar': 30},

    # {'name': 'radar_only_blip_cap_2_mil_blip_v2', 'radar': None, 'radar_name': 'noun',
    #  'radar_legends': ['clip', 'only_blip_cap_2_mil_blip_v2','RB_neg_only_blip_cap_2_mil_blip_v2','rand_both_neg_only_blip_cap_2_mil_blip_v2'], 'eval_radar_ep': 5,
    #  'eval_radar': ['clip', 'only_blip_cap_2_mil_blip_v2','RB_neg_only_blip_cap_2_mil_blip_v2','rand_both_neg_only_blip_cap_2_mil_blip_v2'], 'start_radar': 30},

    # {'name': 'radar', 'radar': None, 'radar_name': 'radar',
    #  'radar_legends': ['symmetric_avg_pos_features','common_batch_pos_avg_pos_features','kl_pos_avg_pos_features','avg_pos_features'], 'eval_radar_ep': 5,
    #  'eval_radar': ['symmetric_avg_pos_features','common_batch_pos_avg_pos_features','kl_pos_avg_pos_features','avg_pos_features'], 'start_radar': 30},


    {'name': 'radar', 'radar': None, 'radar_name': 'radar',
     'radar_legends': ['merged_models_0.2','merged_models_0.3','merged_models_0.4'],
     'eval_radar_ep': 'all',
     'eval_radar': ['merged_models_0.2','merged_models_0.3','merged_models_0.4'],
     'start_radar': 30},

    # 'eval_radar': ['merged_models_0.1', 'merged_models_0.25', 'merged_models_0.5', 'merged_models_0.75',
    #                'merged_models_0.9'],

#     {'name': 'blip_cap_cc3m_lora_1h', 'radar': None, 'radar_name': 'blip_cap_cc3m_lora_1h',
#      'radar_legends': ['clip', 'blip_cap_cc3m_lora_1h','v1_blip_cap_cc3m_lora_1h,v2_blip_cap_cc3m_lora_1h,v3_blip_cap_cc3m_lora_1h,v4_blip_cap_cc3m_lora_1h,v5_blip_cap_cc3m_lora_1h,v6_blip_cap_cc3m_lora_1h,v7_blip_cap_cc3m_lora_1h'
# ], 'eval_radar_ep': 5,
#      'eval_radar': ['clip', 'blip_cap_cc3m_lora_1h','v1_blip_cap_cc3m_lora_1h','v2_blip_cap_cc3m_lora_1h','v3_blip_cap_cc3m_lora_1h','v4_blip_cap_cc3m_lora_1h','v5_blip_cap_cc3m_lora_1h','v6_blip_cap_cc3m_lora_1h','v7_blip_cap_cc3m_lora_1h'
#  ], 'start_radar': 30},


    # {'name': 'ft_cc3m', 'radar': None, 'radar_name': 'fine_tune_cc3m','radar_legends':['CLIP', 'CLIP + LoRA','RB+LLM Negs', 'RB+LLM Negs + Pos', ],'eval_radar_ep':5,
    #  'eval_radar': ['clip', 'cc3m_lora','cc3m_both_negs','cc3m_pos_both_negs',],'start_radar':30},#paper
    #
    # {'name': 'fs_cc3m', 'radar': None, 'radar_name': 'from_scratch_cc3m','radar_legends':['CLIP','RB+LLM Negs', 'RB+LLM Negs + Pos',],'eval_radar_ep':5,
    #  'eval_radar': ['fs_cc3m_baseline','fs_cc3m_both_negs','fs_cc3m_pos_both_negs',],'start_radar':20},#paper

    # {'name': 'ft_cc3m_all_datasets', 'radar': None, 'radar_name': 'ft_cc3m_all_datasets',
    #  'radar_legends': ['CLIP', 'CLIP + LoRA', 'RB+LLM Negs', 'RB+LLM Negs + Pos', ], 'eval_radar_ep': 5,
    #  'eval_radar': ['clip', 'cc3m_lora', 'cc3m_both_negs', 'cc3m_pos_both_negs', ], 'start_radar': 30},  # ALL datasets

    # {'name': 'fs_cc3m_all_datasets', 'radar': None, 'radar_name': 'from_scratch_cc3m_all_datasets','radar_legends':['CLIP','RB+LLM Negs', 'RB+LLM Negs + Pos',],'eval_radar_ep':5,
    #  'eval_radar': ['fs_cc3m_baseline','fs_cc3m_both_negs','fs_cc3m_pos_both_negs',],'start_radar':20},# ALL datasets

    # {'name': 'fs_cc3m_vit16', 'radar': None, 'radar_name': 'from_scratch_cc3m_vit16',
    #  'radar_legends': ['CLIP', 'RB+LLM Negs', 'RB+LLM Negs + Pos', ], 'eval_radar_ep': 5,
    #  'eval_radar': ['fs_cc3m_baseline_vit_16', 'fs_cc3m_both_negs_vit_16', 'fs_cc3m_pos_both_negs_vit_16', ], 'start_radar': 20},#vit 16

    # {'name': 'fine_tune_cc3m_vit16', 'radar': None, 'radar_name': 'fine_tune_cc3m_vit16','model': 'ViT-B/16',
    #  'radar_legends': ['CLIP', 'RB+LLM Negs', 'RB+LLM Negs + Pos', ], 'eval_radar_ep': 5,
    #  'eval_radar':['cc3m_no_training_vit16_assaf','cor_cc3m_both_negs_vit16_assaf','cor_cc3m_pos_both_negs_vit16_assaf'],
    #  'start_radar': 20},  # vit 16



    # {'name': 'ft_laion', 'radar': None, 'radar_name': 'fine_tune_laion','eval_radar_ep':5,'radar_legends':['CLIP', 'CLIP + LoRA', 'RB+LLM Negs', 'RB+LLM Negs + Pos'],
    #  'eval_radar': ['clip','laion_lora','laion_both_negs','laion_pos_both_negs',],'start_radar':50}, #paper




    # {'name': 'cc3m_ep_10', 'radar': None, 'radar_name': 'fine_tune_cc3m','eval_radar_ep':10,
    #  'eval_radar': ['cc3m_lora','cc3m_pos','cc3m_rb_neg','cc3m_auto_neg','cor_cc3m_both_negs','cor_cc3m_pos_both_negs']},

    # {'name': 'ft_cc3m', 'radar': None, 'radar_name': 'fine_tune_cc3m',
    #  'eval_radar': ['clip','cc3m_baseline','cc3m_lora','cc3m_pos','cc3m_rb_neg','cc3m_auto_neg','cc3m_both_negs','cc3m_pos_both_negs']},#

    # {'name': 'fs_laion_ep_5', 'radar': None, 'radar_name': 'from_scratch_laion',
    #  'eval_radar': ['fs_laion_baseline','fs_laion_rb_neg','fs_laion_auto_neg','fs_laion_pos_both_negs']},

    # {'name': 'cyc', 'radar': None, 'radar_name': 'cyc',
    #  'eval_radar': ['cyclip_baseline_lora_openai','cyclip_pos_both_negs_lora_openai','cyclip_baseline','cyclip_rb_neg','cyclip_both_negs']},

]
# import  torch.distributed.launch
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




