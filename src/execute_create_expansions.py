import argparse
from cvar_pyutils.print_utlis import Logger
from cvar_pyutils.ccc import submit_job
from glob import glob
import numpy as np
import pickle

# missing_cc3m_idx = pickle.load( open( "missing_cc3m_idx.p", "rb" ) )
# missing_cc3m_idx = pickle.load( open( "pending_captions_cc3m.p", "rb" ) )
# missing_cc3m_idx = pickle.load( open( "missing_extra_idxs_cc3m.p", "rb" ) )
# missing_cc3m_idx = list(missing_cc3m_idx)

print ("submitting jobs")
number_jobs = 0
for ii in range (0, 1000000, 1000):

    # submit_job(command_to_run=f"export TRANSFORMERS_CACHE=/dccstor/leonidka1/data/cc3m_LLM_outputs/cache;python prompt_cc3m_longer_captions.py {ii}", machine_type='x86', # conda_env='py38',
    #                             duration='24h', num_nodes=1, num_cores=8, num_gpus=1, mem='20g', gpu_type='a100',mail_log_file_when_done='paola@ibm.com')

    # export TRANSFORMERS_CACHE=/dccstor/leonidka1/data/cc3m_LLM_outputs/cache
    submit_job(command_to_run=f"python create_expansions.py --start_idx {ii}", machine_type='x86',
                               duration='6h', num_nodes=1, num_cores=1, num_gpus=1, mem='40g', gpu_type='a100')

    number_jobs += 1

    if number_jobs > 1000:
        break

print ("exit with ii: ", ii)
print ("number_jobs", number_jobs)


