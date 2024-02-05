import argparse
import csv
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from minicons import scorer
import PopulationLM as pop
import os

parser = argparse.ArgumentParser()
parser.add_argument("--exp_path", type = str)
parser.add_argument("--model", default = 'bert-base-uncased', type = str)
parser.add_argument("--model_loc", default = None, type = str)
parser.add_argument("--batch_size", default = 10, type = int)
parser.add_argument("--num_batches", default = -1, type = int)
parser.add_argument("--committee_size", default = 50, type = int)
parser.add_argument("--device", default = "cuda" if torch.cuda.is_available() else "cpu", type = str)
parser.add_argument("--lmtype", default = 'masked', choices = ['mlm', 'masked', 'causal', 'incremental'], type = str)
args = parser.parse_args()

exp_path = args.exp_path
model_name = args.model
model_loc = args.model_loc
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

if model_loc is not None:
    model_id = os.path.abspath(model_loc)
else:
    model_id = model_name

if lm_type == "masked" or lm_type == "mlm":
    transformer = scorer.MaskedLMScorer(model_id, device)
elif lm_type == "incremental" or lm_type == "causal":
    transformer = scorer.IncrementalLMScorer(model_id, device)

if "/" in model_name:
    model_name = model_name.replace("/", "_")

stimuli_loader = DataLoader(dataset, batch_size = batch_size, num_workers=0)

# convert the internal model to use MC Dropout
pop.DropoutUtils.convert_dropouts(transformer.model)
pop.DropoutUtils.activate_mc_dropout(transformer.model, activate=True, random=0.1)

results = []
control_results = []
conclusion_only = []


column_names += ["dv_prob"]
with open(exp_path + f"/results_{model_name}.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(column_names)

# create a lambda function alias for the method that performs classifications
call_me = lambda p1, q1: transformer.conditional_score(p1, q1, reduction=lambda x: (x.sum(0).item(), x.mean(0).item(), x.tolist()))

if num_batches < 0:
    num_batches = len(stimuli_loader)
for batch in tqdm(stimuli_loader):
    out_dataset = [[], [], [], []]
    priming_scores = []
    for i in range(4):
        out_dataset[i].extend(batch[i])
        
    results = {'dv_prob': []}
    p_list = batch[0]
    dv_list = batch[5]
    population = pop.generate_dropout_population(transformer.model, lambda: call_me(p_list, dv_list), committee_size=committee_size)
    outs = [item for item in pop.call_function_with_population(transformer.model, population, lambda: call_me(p_list, dv_list))]

    transposed_outs = [[row[i] for row in outs] for i in range(len(outs[0]))]

    priming_scores = [score for score in transposed_outs]

    results['dv_prob'].extend(priming_scores)

    out_dataset.append(results['dv_prob'])
    with open(exp_path + f"/results_{model_name}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list(zip(*out_dataset)))

print(exp_path + f"/results_{model_name}.csv")
