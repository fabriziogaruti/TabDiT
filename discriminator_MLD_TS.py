import os
from os import makedirs
from os.path import join
from loguru import logger
import numpy as np
import pandas as pd
import torch
import random
import json

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh import extract_features
from tsfresh.feature_selection.relevance import calculate_relevance_table

# GENERAL IMPORTS
from args import define_ablation_parser, change_ablation_config
from testing.test_utils import print_statistics_binary
from dataset.discriminator import DiscriminatorDataset

from configs.config_general import CONFIG_DICT

from sklearn import preprocessing



def extract_ts_features(dataset, categorical_data_usage, use_minimal_feature, dataset_name_to_use):

    data = dataset.dataset_dataframe

    column_names_list = [i for i in data.columns if i != "user_id"]
    cat_cols = [col for col in column_names_list if col not in CONFIG_DICT["regression_fields"][dataset_name_to_use]]
    data[cat_cols] = data[cat_cols].astype(str)

    if categorical_data_usage == "concat":
        cat_data = data[["user_id",] + list(cat_cols)]
        cat_data = cat_data.groupby('user_id').aggregate(list)[cat_cols]
        for cat_col in cat_cols:
            sequence_len = max([len(i) for i in cat_data[cat_col].tolist()])
            flattened_df = pd.DataFrame(cat_data[cat_col].tolist(), index=cat_data.index, columns=[f'{cat_col}{i}' for i in range(sequence_len)])
            flattened_df.fillna(value="None", inplace=True)
            cat_data = pd.concat([cat_data, flattened_df], axis=1)
            cat_data.drop(cat_col, axis=1, inplace=True)
        df_transformed = data.drop(cat_cols, axis=1)
    elif categorical_data_usage == "onehot":
        data_onehot = pd.get_dummies(data[cat_cols], prefix=cat_cols).astype(int)
        data = data.drop(cat_cols, axis=1)
        df_transformed = pd.concat([data, data_onehot], axis=1)
    else:  # categorical_data_usage == "ordinal":
        for cat_col in cat_cols:
            le = preprocessing.LabelEncoder()
            le.fit(data[cat_col])
            data[cat_col] = le.transform(data[cat_col])
            data[cat_col] = data[cat_col].astype(float)
        df_transformed = data
    
    labels = pd.Series(dataset.labels)

    if use_minimal_feature:
        settings = MinimalFCParameters()
    else:
        settings = EfficientFCParameters()
    
    extracted_features = extract_features(df_transformed, column_id="user_id", default_fc_parameters=settings)

    if not use_minimal_feature:
        extracted_features = extracted_features.dropna(axis=1)
        relevance_table = calculate_relevance_table(extracted_features, labels)
        if relevance_table.relevant.sum() != 0:
            relevant_features = relevance_table[relevance_table.relevant].feature
        else:
            logger.error("No relevant features, taking only first 100 features!")
            relevance_table = relevance_table.sort_values(by='p_value', ascending=False)
            relevant_features = relevance_table[:100].feature
        extracted_features = extracted_features.loc[:, relevant_features]

    df_dataset = extracted_features
    cat_cols = None
    if categorical_data_usage == "concat":
        num_features_num = len(df_dataset.columns)  # [i for i in range(18)]
        df_dataset = pd.concat([df_dataset, cat_data], axis=1)
        cat_cols = [c for i, c in enumerate(df_dataset.columns) if i >= num_features_num]
    y = labels.astype(int)
    assert len(y) == len(df_dataset)

    return df_dataset, y, cat_cols



def main():
    dataset_name_to_use = CONFIG_DICT['dataset_name_to_use']
    discriminator_model = CONFIG_DICT['discriminator_model']  # 'catboost'
    categorical_data_usage = CONFIG_DICT['categorical_data_usage']  # 'ordinal'
    use_minimal_feature = CONFIG_DICT['use_minimal_feature']  # True
    use_parent_table = CONFIG_DICT['use_parent_table']

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
    logger.info(f'running RF SEQUENCE discrimination on experiment {data_root}')

    dataset = DiscriminatorDataset(root=data_root,
                                    discriminate_with_parent = use_parent_table,
                                    seq_len_max = CONFIG_DICT['seq_len_max'],)

    df_dataset, y, cat_cols = extract_ts_features(dataset, categorical_data_usage, use_minimal_feature, dataset_name_to_use)



    if use_parent_table:
        if categorical_data_usage != "concat":
            cat_cols = []

        dataset_parent_data = dataset.parent_dataset_dataframe
        for col in dataset_parent_data.columns:
            if col != 'user_id':
                dataset_parent_data[col] = dataset_parent_data[col].astype(np.int64)
        logger.info(f'dataset data before parent merge : shape {df_dataset.shape}:\n{df_dataset.head()}')
        logger.info(f'parent dataset data : shape {dataset_parent_data.shape}:\n{dataset_parent_data.head()}')

        parent_cols = [c for c in dataset_parent_data.columns if c != 'user_id']
        df_dataset['user_id'] = dataset.dataset_dataframe['user_id'].unique()
        df_dataset = pd.merge(df_dataset, dataset_parent_data, on=['user_id'], how='left')
        df_dataset = df_dataset.drop(columns=['user_id',])
        cat_cols += [c for c in df_dataset.columns if c in parent_cols]
        logger.info(f'dataset data after parent merge : shape {df_dataset.shape}:\n{df_dataset.head()}')

    logger.info("Train test split")
    logger.info(f"Seed: {seed}")
    logger.info(f"Test split percentace: {CONFIG_DICT['test_split_percentage']}")
    logger.info(f'final input dataset columns are {list(df_dataset.columns)}')
    logger.info(f'categorical columns are : {cat_cols}')
    idx_categorical_features = [i for i, c in enumerate(df_dataset.columns) if c in cat_cols] if cat_cols is not None else None
    X_train, X_test, y_train, y_test = train_test_split(df_dataset, y, test_size=CONFIG_DICT['test_split_percentage'], random_state=seed)

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
