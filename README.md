# Bill's Seq2Seq Generation Codebase

Table of Contents
=================
  * [Installation](#installation)
  * [Data Preparation](#data-preparation)
  * [Training](#training)
  * [Evaluation](#evaluation)

## Installation
 - Git clone the repo
 - Install conda environment from `environment.yml` (change env name and path).
 - Install the NMT package in `src/OpenNMT-py/`
```angular2html
cd src/OpenNMT-py/
python setup.py install
```
 - Not sure if it works (check the environment by yourself)

## Data Preparation
You need to prepare 4 txt files per run. For each different dataset, I'll create a new folder and put its data as well as the code to generate and preprocess the data into it. There's an example for the drop dataset.

## Training
(The following files mentioned are in `src/OpenNMT-py/onmt/`)
For a task with a single stream of input and output. The only files potentially need to modify are 
 - `train_single.py` (different training pipelines)
 - `models/*` (define new models, see `model.py` for an example of rewrite longt5, and remember to put it in `models/__init__.py` to import later)
 - `model_builder.py` (import and initialize your model, look at the structure of the code and change it as needed)
 - `utils/loss.py` (define new loss for your model)

### training scripts for sbatch or srun
There is an example in `scripts`. Basically only the third to fifth rows of training arguments need to be changed, i.e. from `train_steps` to `trim_size`. 
Most arguments are straightforward. If there's any confusion please contact me.

Normally I'll use random seed 1. Feel free to change it. Remember to make a new directory `slurm_output` if you're using sbatch.

## Evaluation

You have to change nothing except the `load_test_model` function in `model_builder.py` if you're doing seq2seq tasks.

For classification tasks, the scenario is complex. You might change functions in `translate/translator.py`. 
Specifically, you might not call `_translate_batch_with_strategy`, as you should only have the encoder (you can still use the `_run_encoder` function) but not the decoder. And you have to change the log and output file correspondingly.

### evaluation scripts for sbatch or srun
See `scripts/` for examples.

### test other measures after generation
After running the training and evaluation scripts, you should have three folders in `src/OpenNMT-py/` containing relevant information,
`tf_checkpoints/` for model checkpoints, `preds/` for model predictions and `logs/` for logging information during training.

If you want to test on other measures, e.g. F1 or etc. You might want to look at the prediction file and analyze it. For example, in the example `drop_data` folder, we have functions in `generate.py` file for testing F1.

