# README FOR CODE UNDERSTANDING

# CODE REVIEW

## GENERAL

* run_experiments.sh: bash script that contains the commands to launch from terminal the different experiments
* config_general.py: a script with a configuration dict
* discriminator_MLD_TS.py: main script called to perform evaluation
* args.py: script that contains the parser for all the arguments
* testing/test_utils.py: contains script to compute the evaluation metrics

## DATASET

* dataset/discriminator.py: is the main script to create a dataset from the generated tabular time series given in the repo, with the cached argument most of the operation are skipped loading the files saved previously (if it is not the first launch)

