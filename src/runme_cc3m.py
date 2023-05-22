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

base_command = 'pyutils-run -m training.main'


base_params={
        # 'train-data': '/dccstor/aarbelle1/data/ConceptualCaptions/CC3M/train_with_cap.csv',
    # 'batch-size': 128,
    'epochs': 5,
}
experiments = [ # A dict or a tuple of two dicts, the first to expand the base_params dict for run parameters and the second to
                # expand the base_job_params dict for job submision parameters
    {'name': 'LLM', 'lora': 4, 'use_only_quality_captions': None,
     'batch-size': 32, 'mil_gpt_negs': None,
     "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'],
     # 'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/evlk/v2_BLIP_GPT_NEO_text_expander_sen/'},
    'pretrained': 'openai', 'mil_gpt': '../LLM_dense/'},

{'name': 'SAM', 'lora': 4,'use_only_quality_captions': None,
     'batch-size': 32, 'mil_gpt_negs': None,
     "vl_negs": None, 'neg_type': 'rand_both', "auto_neg_types": ['NOUN', 'ADP', 'ADJ', 'VERB'], 'mil_batch': 10,
     'pretrained': 'openai', 'mil_gpt': '../SAM_dense/'},

    # 'pretrained': 'openai', 'mil_gpt': '/dccstor/sivandov1/data/cap_anything_folder/'},
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
        # params['workers'] = 0
    if args.chunks>1:
        params.pop('name')
    for curr_chunk in range(args.chunks):
        if args.chunks > 1:
            job_params['name'] = f'curr_chunk:{curr_chunk}'
        else:
            job_params['name'] = f'{job_params["name"]}'
    #     params['curr_chunk'] = curr_chunk
        cmnd = dict2params_cl(params, base_command)
        if not print_only:
            all_job_ids, all_job_outputs = submit_dependant_jobs(command_to_run=cmnd, **job_params)
            for jid, jout in zip(all_job_ids, all_job_outputs):
                print(f'Submitted job: {jid}:\n\tCommand:{cmnd}\n\tOutput: {jout}')
        else:
            print(f'Job number: {ind}\n{cmnd}')




