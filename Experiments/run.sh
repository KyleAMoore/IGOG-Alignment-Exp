declare -a incrementalmodels=("PKU-Alignment/alpaca-7b-reproduced") # "PKU-Alignment/beaver-7b-v1.0" "gpt2" "gpt2-medium" "distilgpt2")
# declare -a mlmmodels=("bert-base-uncased" "../local_models/debiased_model_bert-base-uncased_gender" "../local_models/debiased_model_bert-base-uncased_race")

echo "Running experiments on Incremental Language Models"

for model in ${incrementalmodels[@]}    
do
    echo "Running experiments for ${model} on dataset: CORE_dative_1500sampled.csv!"
    python ../python/experiment.py --model ${model} --device cpu --lmtype incremental --batchsize 10 --committee_size 50 --exp_path ../data/Exp1-basic
done

# echo "Running experiments on Masked Language Models"

# for model in ${mlmmodels[@]}
# do
#     echo "Running experiments for ${model} on dataset: CORE_dative_1500sampled.csv!"
#     python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --batchsize 10 --committee_size 50 --dataset_name "core-da" --dataset_path ../data/PrimeLM_sampled/CORE_dative_1500sampled.csv
#     echo "Running experiments for ${model} on dataset: CORE_transitive_1500sampled.csv!"
#     python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --batchsize 10 --committee_size 50 --dataset_name "core-tr" --dataset_path ../data/PrimeLM_sampled/CORE_transitive_1500sampled.csv
# done

echo "Complete"
read -p "Press enter to continue"
