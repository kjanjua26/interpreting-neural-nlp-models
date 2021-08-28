#!/bin/bash

#SBATCH -J fL1_controlled # name of the job
#SBATCH -o fL1_controlled.txt # the output file name.
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem 350000MB

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kamranejaz98@gmail.com

module load slurm

model="bert-base-cased"
base_folder="final_experiments/with_acts/with_mask/layer_1" # change the path to your code files here.
DIR="/alt/mt/work/durrani/Causation_Analysis/ProbingClassifiers/POS/Representations/${model}"
data_folder="/alt/mt/tcav/data"

concept_path="${DIR}/wsj.train.conllx.word"
concept_labels="${DIR}/wsj.train.conllx.label"
concept_activations="${data_folder}/${model}/wsj.train.conllx.json"

base_path="${DIR}/wsj.20.test.conllx.word"
base_labels="${DIR}/wsj.20.test.conllx.label"
base_activations="${data_folder}/${model}/wsj.20.test.conllx.json"

output_directory="/alt/mt/tcav/${model}/${base_folder}"

layer_wise_cav_pickle_path="/alt/mt/tcav/${model}/${base_folder}/layer_wise_cavs.pickle"
controlled_layer_wise_cav_path="/alt/mt/tcav/${model}/${base_folder}/controlled_layer_wise_cavs.pickle"

mode_tcav="wm"
word="[MASK]" # for roBERTa, change to <mask> here.
model_type="LR" # for the classifier.
process_mode="1" # 0 is for non-MASK, 1 is for MASK.
use_grad="0" # 0 for acts, 1 for grad.
if_controlled="1" # 0 for no controlled experiment, 1 for controlled experiment as well.
workers=8 # the parallel workers.
runs=50 # the number of runs.
if_rand=1 # if you want to do the t-test.
name="fcontrolled_experiment_L1"

#echo "Extract Activations for Model ${model}!"
#python -u extraction.py -m $model -i $concept_path -o $concept_activations -t "json"
#python -u extraction.py -m $model -i $base_path -o $base_activations -t "json"

echo "Prepare Concepts and Training CAVs!"
python -u prepare_concepts.py -n $name -i $concept_path -l $concept_labels -e $concept_activations -c "NN CD VB MD JJ CC" -o $output_directory -lm $model_type -w $workers -rs $runs -ce $if_controlled -tl "1"

echo "Computing TCAVs!"
python -u compute_tcavs.py -n $name -b $base_activations -c $layer_wise_cav_pickle_path -rc $controlled_layer_wise_cav_path -o $output_directory -bs $base_path -bl $base_labels -w $word -rs $runs -m $model -ir $if_rand -pm $process_mode -g $use_grad