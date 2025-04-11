import os
from os import makedirs
from os import path
from os.path import join
from loguru import logger
import numpy as np
import pandas as pd
import torch
import random
import json

# ADD PARENT DIR TO PATH
import sys
sys.path.append(os.path.abspath('..'))

# GENERAL IMPORTS
from args import define_ablation_parser, change_ablation_config
from testing.test_utils_logistic_detection import print_statistics_ld

# DATASET SPECIFIC IMPORT
from configs.config_general import CONFIG_DICT


def read_dataframes(data_root, join_on='user', use_parent_table=False, limit_test_data=False):
    data_file = join(data_root, f"true_trans.parquet")
    data_real = pd.read_parquet(data_file)
    logger.info(f"read data real: {data_real.shape}")
    data_file = join(data_root, f"generated_trans.parquet")
    data_generated = pd.read_parquet(data_file)
    logger.info(f"read data generated: {data_generated.shape}")
    
    parent_real = None
    parent_generated = None
    if use_parent_table:
        logger.info("reading parent true and pred")
        parent_real_file = join(data_root, f"true_parent.parquet")
        parent_generated_file = join(data_root, f"generated_parent.parquet")
        if path.exists(parent_real_file) and path.exists(parent_generated_file):
            parent_real = pd.read_parquet(parent_real_file)
            parent_generated = pd.read_parquet(parent_generated_file)
        else:
            logger.warning(f'Only the true parent table is present, no prediction')

    logger.info(f"Nr users real: {data_real[join_on].nunique()}, generated: {data_generated[join_on].nunique()}")
    logger.info(f"Test data shape: real: {data_real.shape}, generated: {data_generated.shape}")
    if parent_real is not None and parent_generated is not None:
        logger.info(f"Parent data shape: real: {parent_real.shape}, generated: {parent_generated.shape}")

    if limit_test_data:
        logger.info("Sampling dataset")
        real_user_unique = pd.Series(data_real[join_on].unique()).sample(2000)
        syn_user_unique = pd.Series(data_generated[join_on].unique()).sample(2000)
        data_real = data_real[data_real[join_on].isin(real_user_unique.tolist())]
        data_generated = data_generated[data_generated[join_on].isin(syn_user_unique.tolist())]
        logger.info(f"Nr users real: {data_real[join_on].nunique()}, generated: {data_generated[join_on].nunique()}")
        logger.info(f"Test data shape: real: {data_real.shape}, generated: {data_generated.shape}")
        if parent_real is not None and parent_generated is not None:
            parent_real = parent_real[parent_real[join_on].isin(real_user_unique.tolist())]
            parent_generated = parent_generated[parent_generated[join_on].isin(syn_user_unique.tolist())]
            logger.info(f"Parent data shape: real: {parent_real.shape}, generated: {parent_generated.shape}")

    dataframe_tuple = (data_real, data_generated, parent_real, parent_generated)
    return dataframe_tuple
            

def main():
    join_on = 'user'
    use_parent_table = CONFIG_DICT['use_parent_table']
    
    # set random seeds
    seed = CONFIG_DICT['seed']
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    data_root = CONFIG_DICT['data_root']
    output_dir = join(CONFIG_DICT['output_dir'], CONFIG_DICT['experiment_name'])

    logger.info(f'running LD discrimination on experiment {data_root}')

    dataframe_tuple = read_dataframes(data_root, 
                                      join_on=join_on, 
                                      use_parent_table=use_parent_table,
                                      limit_test_data=False)

    print_statistics_ld(dataframe_tuple, 
                        join_on, 
                        output_dir, 
                        nr_seed_train_ld=1, 
                        nr_seed_split_dataset=1)




if __name__ == "__main__":

    parser = define_ablation_parser()
    args = parser.parse_args()
    change_ablation_config(args, CONFIG_DICT)

    all_parameters = {
        'general_configuration_file': CONFIG_DICT,
        'command_line_args': vars(args),
    }

    output_dir = join(CONFIG_DICT['output_dir'], CONFIG_DICT['experiment_name'])
    args_dir = join(output_dir, "args_train.json")
    log_dir = join(output_dir, "logs")
    logging_file = join(log_dir, 'loguru.txt')
    makedirs(output_dir, exist_ok=True)
    makedirs(log_dir, exist_ok=True)
    logger.add(logging_file, mode='w')
    with open(args_dir, 'w') as f:
        json.dump(all_parameters, f, indent=6)

    main()
