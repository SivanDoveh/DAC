# DAC
## Data Preperations
### Training data
Download Conceptual Captions 3M training and validation splits from https://ai.google.com/research/ConceptualCaptions/download  
After data preperation, place the data in `DAC/CC3M_data/training` and `DAC/CC3M_data/validation`  

First, navigate to the src directory:
```shell script
cd src
```

#### Create quality captions:
```shell script
python3 training/main.py --create_quality_captions --save_data --batch-size 1
```
Quality captions should be in  `DAC/quality_captions/` 

#### Create Dense captions:
```shell script
python3 training/main.py --create_SAM --save_data --batch-size 1
```
SAM dense captions should be in `DAC/SAM_dense/`

```shell script
python3 create_LLM_dense.py
```
LLM dense captions should be in `DAC/LLM_dense/`

### Evaluation data
Prepare vl checklist dataset as describe in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  
Then move the vl dataset to `DAC/vl_datasets/`  
If you followd the instructions correctly you should have the following folders inside vl_datasets: **'hake', 'swig', 'vg'**. 

prepare aro dataset as described in https://github.com/mertyg/vision-language-models-are-bows
Then move the aro dataset to `DAC/aro/` 
## Train with Quality and Dense data


### Run the training script

The model will be saved in `DAC/Outputs/exp_name/checkpoints`

To train a network with quality captions and:
* SAM density:
```shell script
python3 training/main.py --epochs 5 --name SAM --lora 4 --use_only_quality_captions --batch-size 32 --mil_dense_negs --vl_negs --neg_type rand_both --auto_neg_types NOUN ADP ADJ VERB --mil_batch 10 --pretrained openai --mil_dense ../SAM_dense/
```
* LLM density:
```shell script
python3 training/main.py --epochs 5 --name LLM --lora 4 --use_only_quality_captions --batch-size 32 --mil_dense_negs --vl_negs --neg_type rand_both --auto_neg_types NOUN ADP ADJ VERB --mil_batch 10 --pretrained openai --mil_dense ../LLM_dense/
```

## Evaluation
### Run the evaluation script

All vl_checklist jsons will be saved in `DAC/eval_jsons/clip/exp_name/` and the result will be printed. 
To prepare the vl checklist evaluate results for the experiment **exp_name** run the following command:
```shell script
python3 training/main.py  --lora 4 --pretrained openai --eval_vl_cklist --eval_only --resume /path/to/checkpoint
```

To print the aro evaluate results for the experiment **exp_name** run the following command:
```shell script
python3 aro_clip_lora_eval.py  --lora 4 --resume /path/to/checkpoint
```
