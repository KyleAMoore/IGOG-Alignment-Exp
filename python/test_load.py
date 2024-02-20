'''
Used to load and run debiased BERT models (by overwriting after loading minicons).
TODO: Transfer functionality to experiment.py and delete this script
'''

import argparse
import csv
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM

from minicons import scorer
import PopulationLM as pop
import os


exp_paths = [
             '../data/Exp1-basic',
             '../data/Exp2-DG',
             '../data/Exp3-PGG',
             '../data/Exp4-CYD',
             '../data/Exp5-FAA',
             '../data/Exp6-WM'
             ]
base_model_name = 'bert-base-uncased'
model_loc = '../local_models/'
batch_size = 10
num_batches = -1
committee_size = 50
device = 'cuda'
lm_type = 'masked'

overwrite_models = [
        # 'debiased_model_bert-base-uncased_gender',
        'debiased_model_bert-base-uncased_race'
    ]

for overwrite_modeL_name in overwrite_models:
    overwrite_model_name = overwrite_models[0]
    for exp_path in exp_paths:
        dataset = []
        with open(exp_path + '/prompts.csv', "r") as f:
            reader = csv.DictReader(f, delimiter='|')
            column_names = reader.fieldnames
            for row in reader:
                dataset.append(list(row.values()))

        if lm_type == "masked" or lm_type == "mlm":
            transformer = scorer.MaskedLMScorer(base_model_name, device)
        elif lm_type == "incremental" or lm_type == "causal":
            transformer = scorer.IncrementalLMScorer(base_model_name, device)

        overwrite_model = AutoModelForMaskedLM.from_pretrained(model_loc+overwrite_model_name+'/', local_files_only=True)
        overwrite_model.to(device)

        transformer.model = overwrite_model

        if "/" in overwrite_model_name:
            overwrite_model_name = overwrite_model_name.replace("/", "_")


        # convert the internal model to use MC Dropout
        pop.DropoutUtils.convert_dropouts(transformer.model)
        pop.DropoutUtils.activate_mc_dropout(transformer.model, activate=True, random=0.1)

        results = []
        control_results = []
        conclusion_only = []

        column_names += ["dv_prob"]
        with open(exp_path + f"/results_{overwrite_model_name}.csv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(column_names)

        # create a lambda function alias for the method that performs classifications
        call_me = lambda p1, q1: transformer.conditional_score(p1, q1, reduction=lambda x: (x.sum(0).item(), x.mean(0).item(), x.tolist()))

        stimuli_loader = DataLoader(dataset, batch_size = batch_size, num_workers=0)
        if num_batches < 0:
            num_batches = len(stimuli_loader)
        for batch in tqdm(stimuli_loader):
            out_dataset = [[] for _ in range(len(batch))]
            priming_scores = []
            for i in range(len(batch)):
                out_dataset[i].extend(batch[i])

            results = {'dv_prob': []}
            p_list = list(batch[0])
            dv_list = list(batch[5])

            population = pop.generate_dropout_population(transformer.model, lambda: call_me(p_list, dv_list), committee_size=committee_size)
            outs = [item for item in pop.call_function_with_population(transformer.model, population, lambda: call_me(p_list, dv_list))]

            transposed_outs = [[row[i] for row in outs] for i in range(len(outs[0]))]

            priming_scores = [score for score in transposed_outs]

            results['dv_prob'].extend(priming_scores)

            out_dataset.append(results['dv_prob'])
            with open(exp_path + f"/results_{overwrite_model_name}.csv", "a", newline='') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerows(list(zip(*out_dataset)))


        print(exp_path + f"/results_{overwrite_model_name}.csv")
