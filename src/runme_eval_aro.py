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
    'duration': '24h',
    'gpu_type': 'a100',# || v100 && hname!=cccxc547 && hname!=cccxc572' ,
}
# base_command = 'python -m cvar_pyutils.pytorch_utils.launch -m training.main'
# base_command = 'pyutils-launch -m training.main'
base_command = 'pyutils-run -m aro_clip_lora_eval'


base_params={
}


experiments = [ ]
# eval_epoch = 5
eval_epoch = 'latest'
ckpoint=f'checkpoints/epoch_{eval_epoch}.pt'
root_ck = '/dccstor/sivandov1/dev/open_clip_vl/Outputs'
# root_ck = '/dccstor/alfassy/dev/open-clip/Outputs/'

models=[]

#cvpr paper
# models.extend(['cc3m_rb_neg','cor_cc3m_both_negs'])
# blip captions
# models.extend(['only_blip_cap_2','RB_neg_only_blip_cap_2','rand_both_neg_only_blip_cap_2','use_extra_blip_cap_expanders'])
# models.extend(['symmetric_avg_pos_features','common_batch_pos_avg_pos_features','kl_pos_avg_pos_features','avg_pos_features'])
# models.extend(['eye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen', 'mb5b32_cap_any_neg_neg_mil', 'cap_any_eye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen', 'n_eye_neg_b_32_mb_5_vl_and_neg_mil_gpt_v2_sen', 'weye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen', 'weye_neg_b_32_mb_5_vl_and_neg_mil_gpt_v2_sen', 'help_eye_neg_b_32_mb_5_vl_and_neg_mil_gpt_v2_sen', 'help3_cap_any_eye_neg_mb_2_vl_and_neg_mil_gpt_v2_sen'])#neg_mil_gpt_v2_sen
# models.extend(['merge_llm_dense_cap_alpha_0.1','merge_llm_dense_cap_alpha_0.25','merge_llm_dense_cap_alpha_0.5','merge_llm_dense_cap_alpha_0.75','merge_llm_dense_cap_alpha_0.9',
# ])
models.extend(['ones_neg_both_use_extra_blip_cap_expanders'])
# models.extend(['avg6_use_v2_extra_blip_expanders_additional_data','max6_use_v2_extra_blip_expanders_additional_data'])



for i,m in enumerate(models):
    path = os.path.join(root_ck,m,ckpoint)
    experiments.append({'name':'eval_aro_'+ str(i),'resume':path,'lora':4})


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




