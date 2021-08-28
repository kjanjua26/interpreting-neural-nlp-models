# Interpreting Deep NLP Models using Concept Activation Vectors
This is the repository for our paper "Interpreting Deep NLP Models using Concept Activation Vectors". The repository contains the code to implement the methodology and also to re-produce the results. We propose a plug-and-play methodology to analyze representations in deep NN models. 

## Train/Run

There are several configurations the models can be run and interepreted with; they are listed below.

1. With activations, with [MASK]
2. With activations, without [MASK]
3. With gradients, without [MASK]
4. With gradients, with [MASK]

We currently experiment with activations and gradients of the trained language model (BERT, XLNet). Our goal is to be able to associate the learned representations to concepts defined at a higher level.

The code base is in the `code/` folder. The configuration is defined in the `code/run.sh` file and can be altered to run any of the listed experiments.

### `code/run.sh` File Explanation

The file `code/run.sh` is divided into two parts. The first part consists of all the paths of extracted activations from all the layers of the trained models.

```
model="bert-base-cased" # this is where the model is defined

base_folder="final_experiments/with_acts/with_mask/layer_1" # change the path to your code files here. For every layer, a new folder is constructed.

DIR="/alt/mt/work/durrani/Causation_Analysis/ProbingClassifiers/POS/Representations/${model}" # this is the base directory containing the representations

data_folder="/alt/mt/tcav/data" # the folder that has the dataset

concept_path="${DIR}/wsj.train.conllx.word" # the concept sents
concept_labels="${DIR}/wsj.train.conllx.label" # the concept labels
concept_activations="${data_folder}/${model}/wsj.train.conllx.json" # the concept activations

base_path="${DIR}/wsj.20.test.conllx.word" # the base sents
base_labels="${DIR}/wsj.20.test.conllx.label" # the base labels
base_activations="${data_folder}/${model}/wsj.20.test.conllx.json" # the base activations

output_directory="/alt/mt/tcav/${model}/${base_folder}" # the directory to store the pickle files in 

layer_wise_cav_pickle_path="/alt/mt/tcav/${model}/${base_folder}/layer_wise_cavs.pickle" # the path to store layer wise CAV pickle file in
controlled_layer_wise_cav_path="/alt/mt/tcav/${model}/${base_folder}/controlled_layer_wise_cavs.pickle" # the path to store controlled layer wise CAV pickle file
```

The second part is related to the model configurations.

```
mode_tcav="wm" # this it the mode, wm -> word mask
word="[MASK]" # the type of mask [MASK] for BERT, for roBERTa, change to <mask> here.
model_type="LR" # for the classifier.
process_mode="1" # 0 is for non-MASK, 1 is for MASK.
use_grad="0" # 0 for acts, 1 for grad.
if_controlled="1" # 0 for no controlled experiment, 1 for controlled experiment as well.
workers=8 # the parallel workers.
runs=50 # the number of runs.
if_rand=1 # if you want to do the t-test.
name="fcontrolled_experiment_L1"
```

Once the experiment is complete, there should be several files in the folder path you set in `base_folder` in `code/run.sh`. The results are contained in the file named: `inference_word_mode_masked_test.pickle`.

## Inference

Once `inference_word_mode_masked_test.pickle` has been generated, pass this to the `inference/Inference_Plots_TCAV.ipynb` and the TCAV matrices will be displayed showing TCAV score for each test and [MASK] concept.