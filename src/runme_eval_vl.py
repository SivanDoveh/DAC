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
    'num_nodes': 1,
    'num_gpus': 1,
    'num_cores': 32,
    'mem': '32g',
    'duration': '12h',
    'gpu_type': 'a100',# || v100 && hname!=cccxc547 && hname!=cccxc572' ,
}
# base_command = 'python -m cvar_pyutils.pytorch_utils.launch -m training.main'
# base_command = 'pyutils-launch -m training.main'
base_command = '/dccstor/sivandov1/anaconda3/envs/vl/bin/pyutils-run -m training.main'


base_params={
    'train-data': '/dccstor/aarbelle1/data/ConceptualCaptions/CC3M/train_with_cap.csv',
    'val-data': "/dccstor/aarbelle1/data/ConceptualCaptions/CC3M/val_with_cap.csv",
    "eval_recall": None,
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
    'workers': 32,
    "beta1": 0.9,
    "beta2": 0.98,
    "eps": 1.0e-6,
    'pretrained':'openai',
    'chunks': args.chunks,
    'curr_chunk': 0,
    "eval_vl_cklist":None,
    "save-most-recent":None,
    "eval_vl_cklist_all": None,
    "eval_only":None,
    "zeroshot-frequency":1,

    # "save_eval_model":None,
}


experiments = [ ]
# eval_epoch = 5
eval_epoch = 'latest'
ckpoint=f'checkpoints/epoch_{eval_epoch}.pt'
root_ck = '/dccstor/sivandov1/dev/open_clip_vl/Outputs'


# models=['merge_llm_dense_cap_alpha_0.1','merge_llm_dense_cap_alpha_0.25','merge_llm_dense_cap_alpha_0.5','merge_llm_dense_cap_alpha_0.75','merge_llm_dense_cap_alpha_0.9']
models=['cap_any_mb10_no_milneg']

# models=['dg_v2_12mb5_16']

#cc3m
#ft:
# models.extend(['clip','cc3m_lora','cc3m_pos','cc3m_rb_neg','cc3m_auto_neg','cor_cc3m_both_negs','cor_cc3m_pos_both_negs'])#paper
# models.extend(['blip_cap_cc3m_lora_1h','v8_only_blip_cap_1h'])
# models.extend(['only_blip_cap_2','RB_neg_only_blip_cap_2','rand_both_neg_only_blip_cap_2'])
# models.extend(['lr1_RB_neg_only_blip_cap_2','lr2_RB_neg_only_blip_cap_2','lr3_RB_neg_only_blip_cap_2'])
# models.extend(['symmetric_avg_pos_features','common_batch_pos_avg_pos_features','kl_pos_avg_pos_features','avg_pos_features'])

# models.extend(['eye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen', 'mb5b32_cap_any_neg_neg_mil', 'cap_any_eye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen', 'n_eye_neg_b_32_mb_5_vl_and_neg_mil_gpt_v2_sen', 'weye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen', 'weye_neg_b_32_mb_5_vl_and_neg_mil_gpt_v2_sen', 'help_eye_neg_b_32_mb_5_vl_and_neg_mil_gpt_v2_sen', 'help3_cap_any_eye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen'])
# models.extend(['sivangl_cap_any_mb2012'])
# models10=['cor_cc3m_both_negs','cc3m_auto_neg']#['cc3m_lora']#,'cc3m_pos','cc3m_rb_neg','cc3m_auto_neg','cor_cc3m_both_negs','cor_cc3m_pos_both_negs']#10
#fs
# models.extend(['Leo_fs_cc3m_baseline','fs_cc3m_pos','fs_cc3m_rb_neg','Leo_fs_cc3m_auto_neg','Leo_fs_cc3m_both_negs','Leo_fs_cc3m_pos_both_negs'])#paper


#lorot
# models=['K_lora_6_lr_10_cc3m_pos_both_negs','K_lora_6_lr_8_cc3m_pos_both_negs'] #needs zs1,5
# models=['s_ora_2_lr_8_cc3m_pos_b']

#LAION
#ft
# models.extend(['laion_lora','laion_pos','A_laion_pos','laion_rb_neg','R_laion_both_negs','A_laion_pos_both_negs',])#R_laion_auto_neg
# models.extend(['laion_baseline'])

#fs
# models.extend(['fs_laion_baseline','fs_laion_rb_neg','fs_laion_auto_neg','fs_laion_pos_both_negs'])#new_fs_laion_both_negs,fs_laion_pos
# models=['neg_w_0_no_seperate_neg_loss_pos']

# 'lora':4,

#cc3m vit 16
#fs
# 'fs_cc3m_both_negs_vit_16','fs_cc3m_pos_both_negs_vit_16','fs_cc3m_baseline_vit_16'
#ft
#cc3m_no_training_vit16_assaf,'cc3m_baseline_vit16_assaf','cor_cc3m_both_negs_vit16_assaf','cor_cc3m_pos_both_negs_vit16_assaf'

# models.extend(['cc3m_baseline_vit16_assaf'])

for i,m in enumerate(models):
    path = os.path.join(root_ck,m,ckpoint)
    # experiments.append({'name':'eval_'+ str(i),'resume':path,'model': 'ViT-B/16','lora':4,})
    # experiments.append({'name':'eval_'+ str(i),'resume':path,'lora':4,'kqv_lora':None})
    experiments.append({'name':'eval_'+ str(i),'resume':path,'lora':4})


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
        params['workers'] = 0
        params['debug'] = None
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




