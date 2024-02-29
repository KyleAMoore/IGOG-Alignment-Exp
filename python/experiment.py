'''
Also implemented with insubstantial changes in the below notebook to take advantage of Google Colab compute resources. Will add notebook directly to repo at a later date

https://colab.research.google.com/drive/1h9sZ8DdAMU_MufHsb2D7sNUmelBdX3o7?usp=sharing
'''

import argparse
import csv
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM

from minicons import scorer
import PopulationLM as pop
import os

parser = argparse.ArgumentParser()
parser.add_argument("--exp_path", type = str)
parser.add_argument("--base_model", default = 'bert-base-uncased', type = str)
parser.add_argument("--ow_model_loc", default = None, type = str)
parser.add_argument("--res_loc", default = None, type = str)
parser.add_argument("--batch_size", default = 10, type = int)
parser.add_argument("--num_batches", default = -1, type = int)
parser.add_argument("--committee_size", default = 50, type = int)
parser.add_argument("--device", default = "cuda" if torch.cuda.is_available() else "cpu", type = str)
parser.add_argument("--lmtype", default = 'masked', choices = ['mlm', 'masked', 'causal', 'incremental'], type = str)
args = parser.parse_args()

exp_path = args.exp_path
base_model_name = args.base_model
ow_model_loc = args.ow_model_loc
results_loc = args.res_loc
batch_size = args.batch_size
num_batches = args.num_batches
committee_size = args.committee_size
device = args.device
lm_type = args.lmtype


dataset = []
with open(exp_path + '/prompts.csv', "r") as f:
    reader = csv.DictReader(f, delimiter='|')
    column_names = reader.fieldnames
    for row in reader:
        dataset.append(list(row.values()))

# Load the model
if lm_type == "masked" or lm_type == "mlm":
    transformer = scorer.MaskedLMScorer(base_model_name, device)
elif lm_type == "incremental" or lm_type == "causal":
    transformer = scorer.IncrementalLMScorer(base_model_name, device)

#Overwrite local model with base model (handles local loading limitation in minicons)
if ow_model_loc is not None:
    model_name = os.path.basename(os.path.normpath(ow_model_loc))
    if lm_type in ['mlm', 'masked']:
        overwrite_model = AutoModelForMaskedLM.from_pretrained(ow_model_loc, local_files_only=True)
    else:
        overwrite_model = AutoModelForCausalLM.from_pretrained(ow_model_loc, local_files_only=True)
    overwrite_model.to(device)
    transformer.model = overwrite_model
else:
    model_name = base_model_name

if "/" in model_name:
    model_name = model_name.replace("/", "_")

if results_loc is not None:
    results_loc = exp_path + '/' + results_loc
else:
    results_loc = exp_path + f"/results_{model_name}.csv"

# convert the internal model to use MC Dropout
pop.DropoutUtils.convert_dropouts(transformer.model)
pop.DropoutUtils.activate_mc_dropout(transformer.model, activate=True, random=0.1)

results = []
control_results = []
conclusion_only = []

column_names += ["dv_prob"]
with open(results_loc, "w", newline='') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(column_names)

# create a lambda function alias for the method that performs classifications
call_me = lambda p1, q1: transformer.conditional_score(p1, q1, reduction=lambda x: (x.sum(0).item(), x.mean(0).item(), x.tolist()))

stimuli_loader = DataLoader(dataset, batch_size = batch_size, num_workers=0)
if num_batches < 0:
    num_batches = len(stimuli_loader)
for batch in tqdm(stimuli_loader):
    out_dataset = [[] for _ in range(len(batch))]
    dv_scores = []
    for i in range(len(batch)):
        out_dataset[i].extend(batch[i])

    results = {'dv_prob': []}
    p_list = list(batch[0])
    dv_list = list(batch[5])

    population = pop.generate_dropout_population(transformer.model, lambda: call_me(p_list, dv_list), committee_size=committee_size)
    outs = [item for item in pop.call_function_with_population(transformer.model, population, lambda: call_me(p_list, dv_list))]

    transposed_outs = [[row[i] for row in outs] for i in range(len(outs[0]))]

    dv_scores = [score for score in transposed_outs]

    results['dv_prob'].extend(dv_scores)

    out_dataset.append(results['dv_prob'])
    with open(results_loc, "a", newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(list(zip(*out_dataset)))

print('Results saved to: ', results_loc)