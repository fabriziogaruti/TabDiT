import os
from os import makedirs
from os.path import join
from loguru import logger
import numpy as np
import torch
import random
import json

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from args import define_ablation_parser, change_ablation_config
from testing.test_utils import print_statistics_binary
from dataset.efficacy import EfficacyDataset

from configs.config_general import CONFIG_DICT

from discriminator_MLD_TS import extract_ts_features



def main():
    use_generated_data_efficacy = CONFIG_DICT['use_generated_data_efficacy']
    dataset_name_to_use = CONFIG_DICT['dataset_name_to_use']
    discriminator_model = CONFIG_DICT['discriminator_model']  # 'catboost'
    categorical_data_usage = CONFIG_DICT['categorical_data_usage']  # 'ordinal'
    use_minimal_feature = CONFIG_DICT['use_minimal_feature']  # True

    # set random seeds
    seed = CONFIG_DICT['seed']
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    data_root = CONFIG_DICT['data_root']
    assert os.path.exists(data_root), "Error! Impossible to find generation folder"

    output_dir = join(CONFIG_DICT['output_dir'], CONFIG_DICT['experiment_name'])
    makedirs(output_dir, exist_ok=True)
    logger.info(f'running ML Efficacy on experiment {data_root}')

    dataset = EfficacyDataset(root=data_root,
                              dataset_name_to_use=dataset_name_to_use,
                              seq_len_max = CONFIG_DICT['seq_len_max'],
                              use_generated_data=False,)

    df_dataset, y, cat_cols = extract_ts_features(dataset, categorical_data_usage, use_minimal_feature, dataset_name_to_use)

    logger.info("Train test split")
    logger.info(f"Seed: {seed}")
    logger.info(f"Test split percentace: {CONFIG_DICT['test_split_percentage']}")
    logger.info(f'final input dataset columns are {list(df_dataset.columns)}')
    logger.info(f'categorical columns are : {cat_cols}')
    idx_categorical_features = [i for i, c in enumerate(df_dataset.columns) if c in cat_cols] if cat_cols is not None else None
    X_train, X_test, y_train, y_test = train_test_split(df_dataset, y, test_size=CONFIG_DICT['test_split_percentage'], random_state=seed)


    if use_generated_data_efficacy:
        logger.info("USE GENERATED DATA FOR EFFICACY")
        X_train, y_train = None, None
        dataset_generated = EfficacyDataset(root=data_root,
                            dataset_name_to_use=dataset_name_to_use,
                            seq_len_max = CONFIG_DICT['seq_len_max'],
                            use_generated_data=True,)
        df_dataset_gen, y_gen, cat_cols = extract_ts_features(dataset_generated, categorical_data_usage, use_minimal_feature, dataset_name_to_use)

        logger.info("Train test split")
        logger.info(f"Seed: {seed}")
        logger.info(f"Test split percentace: {CONFIG_DICT['test_split_percentage']}")
        logger.info(f'final input dataset columns are {list(df_dataset_gen.columns)}')
        logger.info(f'categorical columns are : {cat_cols}')
        idx_categorical_features = [i for i, c in enumerate(df_dataset_gen.columns) if c in cat_cols] if cat_cols is not None else None

        X_train, _, y_train, _ = train_test_split(df_dataset_gen, y_gen, test_size=CONFIG_DICT['test_split_percentage'], random_state=seed)
    else:
        logger.info("USE REAL DATA FOR EFFICACY")

    estimators = [10,15,20]
    depths = [10,15]
    best_score = 0

    X_train_part, X_eval, y_train_part, y_eval = train_test_split(X_train, y_train, test_size=CONFIG_DICT['test_split_percentage'], random_state=seed)

    train_part_data = Pool(data=X_train_part, label=y_train_part, cat_features=idx_categorical_features)  # cat_features=[0, 1, 2]
    eval_data = Pool(data=X_eval, cat_features=idx_categorical_features)
    
    for n_est in estimators:
        for depth in depths:
            if discriminator_model == 'catboost':
                model = CatBoostClassifier(n_estimators=n_est, max_depth=depth,)
                model.fit(train_part_data)
                preds = model.predict(eval_data)
            else:
                model = XGBClassifier(n_estimators=n_est, max_depth=depth,)
                model.fit(X_train_part, y_train_part)
                preds = model.predict(X_eval)
            
            score = accuracy_score(y_eval, preds)
            logger.info(f'with parameters ({n_est}, {depth}) , Accuracy is {score:.3f}')
            if score > best_score:
                best_params = (n_est, depth)
                best_score = score
    logger.debug(f'best parameters are {best_params} , best Accuracy is {best_score:.3f}')

    train_data = Pool(data=X_train, label=y_train, cat_features=idx_categorical_features)  # cat_features=[0, 1, 2]
    test_data = Pool(data=X_test, cat_features=idx_categorical_features)

    if discriminator_model == 'catboost':
        logger.info(f'Fitting CatBoostClassifier... \n')
        model = CatBoostClassifier(n_estimators=best_params[0], max_depth=best_params[1],)
        model.fit(train_data)
        y_pred = model.predict_proba(test_data)
    else:
        logger.info(f'Fitting XGBClassifier... \n')
        model = XGBClassifier(n_estimators=best_params[0], max_depth=best_params[1],)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)

    y_pred = y_pred[:,1]
    print_statistics_binary(y_pred, y_test, output_dir, log=True)



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
