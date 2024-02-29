declare -a exppaths=("../data/Exp1-basic" "../data/Exp2-DG" "../data/Exp3-PGG" "../data/Exp4-CYD" "../data/Exp5-FAA" "../data/Exp6-WM")

declare -a incrementalmodels=("PKU-Alignment/alpaca-7b-reproduced" "PKU-Alignment/beaver-7b-v1.0")
declare -a mlmmodels=("bert-base-uncased") 

declare -a basemodels=("bert-base-uncased" "bert-base-uncased")
declare -a overwritemodels=("../local_models/debiased_model_bert-base-uncased_gender" "../local_models/debiased_model_bert-base-uncased_race")
declare -a owmodeltypes=("masked" "masked")


echo "Running experiments on Masked Language Models"
for model in ${mlmmodels[@]}    
do
    for exppath in ${exppaths[@]}
    do
        echo "Running experiments for ${model} on dataset: ${exppath}"
        python ../python/experiment.py --base_model ${model} --device cuda --lmtype masked --batch_size 10 --committee_size 50 --exp_path ${exppath}
    done
done

echo "Running experiments on Incremental Language Models"
for model in ${incrementalmodels[@]}    
do
    for exppath in ${exppaths[@]}
    do
        echo "Running experiments for ${model} on dataset: ${exppath}"
        python ../python/experiment.py --base_model ${model} --device cuda --lmtype incremental --batch_size 10 --committee_size 50 --exp_path ${exppath}
    done
done

echo "Running experiments on local models"
for i in ${!overwritemodels[@]}
do
    for exppath in ${exppaths[@]}
    do
        python ../python/experiment.py --base_model ${basemodels[$i]} --ow_model_loc ${overwritemodels[$i]} --device cuda --lmtype ${owmodeltypes[$i]} --batch_size 10 --committee_size 50 --exp_path ${exppath}
    done
done

read -p "Experiments complete. Press enter to close..."
