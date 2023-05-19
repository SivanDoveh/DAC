import argparse
from cvar_pyutils.print_utlis import Logger
from cvar_pyutils.ccc import submit_job
from glob import glob
import numpy as np
import pickle

missing_cc3m_idx = pickle.load( open( "/dccstor/leonidka1/data/cc3m_LLM_outputs/blip2_positives_cc3m_gpt_extra/all_files_in_blip2_positive_cc3m_folder.p", "rb" ) )

number_jobs = 0
# 400100
# 1400100
# 2860100 (next) -- use 1500 batch
# for ii in range (2860100, len(missing_cc3m_idx), 1000):
for ii in range(0, 1000000, 1000):
# for ii in range(0, 1000, 1000):

    # export TRANSFORMERS_CACHE=/dccstor/leonidka1/data/cc3m_LLM_outputs/cache
    submit_job(command_to_run=f"python create_blip_expansions.py --start_idx {ii}", machine_type='x86',
                               duration='1h', num_nodes=1, num_cores=1, num_gpus=1, mem='40g', gpu_type='a100')
    number_jobs += 1

    if number_jobs > 1000:
        break

print ("exit with ii: ", ii)