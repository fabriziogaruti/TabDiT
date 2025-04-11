
CONFIG_DICT = {
    'dataset_name_to_use': 'age1',
    'seed': 9,

    'data_root': './generated_data/age1/uncond_s0',
    'output_dir': './discriminator_results/age1/uncond_s0',
    'experiment_name': 'discriminator_multi_CB',
    
    'discriminator_model': 'catboost',
    'categorical_data_usage': 'ordinal',
    'use_minimal_feature': True,

    'use_parent_table': False,

    'test_split_percentage': 0.20,
    'seq_len_max': -1,

    'use_generated_data_efficacy': True,

    'regression_fields': {
        'age1': ['amount_rur'],
        'age2': ['transaction_amt'],
        'airbnb': ['secs_elapsed'],
        'pkdd99': ['amount_trans', 'balance'],
        'leaving': ['amount'],
        'rossmann': ['Customers', 'Sales'],
    }
}


