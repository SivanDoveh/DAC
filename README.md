# DAC
## Data Preperations
### Training data
Download Conceptual Captions 3M training and validation splits from https://ai.google.com/research/ConceptualCaptions/download  
After data preperation, place the data in `DAC/CC3M_data/training` and `DAC/CC3M_data/validation`  

#### Train with Quality and Dense data
Unzip and Place the quality captions in  `DAC/quality_captions/`  and the dense captions in `DAC/SAM_dense/` and `DAC/LLM_dense/`

### Evaluation data
Prepare vl checklist dataset as describe in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  
Then move the vl dataset to `DAC/vl_datasets/`  
If you followd the instructions correctly you should have the following folders inside vl_datasets: **'hake', 'swig', 'vg'**. 

## Training

### Run the training script
First, navigate to the src directory:
```shell script
cd src
```
The model will be saved in `TSVLC/Outputs/exp_name/checkpoints`

To train a network with:
* RB negative generation:
```shell script
python3 training/main.py --name exp_name --vl_negs --lora 4 --neg_type rule_based --pretrained openai
```
