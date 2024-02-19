declare -a incrementalmodels=("PKU-Alignment/alpaca-7b-reproduced" "PKU-Alignment/beaver-7b-v1.0") #"gpt2" "gpt2-medium" "distilgpt2")
declare -a mlmmodels=("bert-base-uncased" "../local_models/debiased_model_bert-base-uncased_gender" "../local_models/debiased_model_bert-base-uncased_race")

echo "Running experiments on Masked Language Models"
for model in ${mlmmodels[@]}    
do
    echo "Running experiments for ${model} on dataset: Exp1-basic"
    python ../python/experiment.py --model ${model} --device cuda --lmtype masked --batch_size 10 --committee_size 50 --exp_path ../data/Exp1-basic
    python ../python/experiment.py --model ${model} --device cuda --lmtype masked --batch_size 10 --committee_size 50 --exp_path ../data/Exp2-DG
    python ../python/experiment.py --model ${model} --device cuda --lmtype masked --batch_size 10 --committee_size 50 --exp_path ../data/Exp3-PGG
    python ../python/experiment.py --model ${model} --device cuda --lmtype masked --batch_size 10 --committee_size 50 --exp_path ../data/Exp4-CYD
    python ../python/experiment.py --model ${model} --device cuda --lmtype masked --batch_size 10 --committee_size 50 --exp_path ../data/Exp5-FAA
    python ../python/experiment.py --model ${model} --device cuda --lmtype masked --batch_size 10 --committee_size 50 --exp_path ../data/Exp6-WM
done

echo "Running experiments on Incremental Language Models"
for model in ${incrementalmodels[@]}    
do
    echo "Running experiments for ${model} on dataset: Exp1-basic"
    python ../python/experiment.py --model ${model} --device cuda --lmtype incremental --batch_size 10 --committee_size 50 --exp_path ../data/Exp1-basic
    python ../python/experiment.py --model ${model} --device cuda --lmtype incremental --batch_size 10 --committee_size 50 --exp_path ../data/Exp2-DG
    python ../python/experiment.py --model ${model} --device cuda --lmtype incremental --batch_size 10 --committee_size 50 --exp_path ../data/Exp3-PGG
    python ../python/experiment.py --model ${model} --device cuda --lmtype incremental --batch_size 10 --committee_size 50 --exp_path ../data/Exp4-CYD
    python ../python/experiment.py --model ${model} --device cuda --lmtype incremental --batch_size 10 --committee_size 50 --exp_path ../data/Exp5-FAA
    python ../python/experiment.py --model ${model} --device cuda --lmtype incremental --batch_size 10 --committee_size 50 --exp_path ../data/Exp6-WM
done

read -p "Experiments complete. Press enter to close..."
