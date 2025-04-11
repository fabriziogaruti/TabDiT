import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def define_ablation_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name_to_use", type=str,
                        default=None,
                        help='dataset_name_to_use')
    parser.add_argument("--seed", type=int,
                        default=None,
                        help='seed')

    parser.add_argument("--data_root", type=str,
                        default=None,
                        help='root directory for dataset csv files')
    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help='directory containing all results')
    parser.add_argument("--experiment_name", type=str,
                        default=None,
                        help='folder containing this experiment results')
    
    parser.add_argument("--discriminator_model", type=str,
                        default=None,
                        help="string to define the ML model, can be: \'catboost\', \'xgboost\'")
    parser.add_argument("--categorical_data_usage", type=str,
                        default=None,
                        help="how to use categorical data, can be \'concat\', \'onehot\', \'ordinal\'. \'concat\' means to concatenate the sequence of categories to the extracted features.")
    parser.add_argument("--use_minimal_feature", type=str2bool,
                        default=None,
                        help="whether to extract minimal (or complete) features with tsfresh library")

    parser.add_argument("--use_parent_table", type=str2bool, 
                        default=None, 
                        help='whether to use parent data')

    parser.add_argument("--test_split_percentage", type=float,
                        default=None,
                        help="train test split percentage, default to 0.2")
    parser.add_argument("--seq_len_max", type=int,
                            default=None,
                            help='maximum length of the time serie sequence of inputs')
    
    parser.add_argument("--use_generated_data_efficacy", type=str2bool,
                        default=None,
                        help="Whether to use the generated data to train the ML efficacy model. Otherwise use the real data.")
    return parser



def change_ablation_config(args, configs):
    if args.dataset_name_to_use is not None:
        configs['dataset_name_to_use'] = args.dataset_name_to_use
    if args.seed is not None:
        configs['seed'] = args.seed

    if args.data_root is not None:
        configs['data_root'] = args.data_root
    if args.output_dir is not None:
        configs['output_dir'] = args.output_dir
    if args.experiment_name is not None:
        configs['experiment_name'] = args.experiment_name 

    if args.discriminator_model is not None:
        configs['discriminator_model'] = args.discriminator_model
    if args.categorical_data_usage is not None:
        configs['categorical_data_usage'] = args.categorical_data_usage
    if args.use_minimal_feature is not None:
        configs['use_minimal_feature'] = args.use_minimal_feature

    if args.use_parent_table is not None:
        configs['use_parent_table'] = args.use_parent_table

    if args.test_split_percentage is not None:
        configs['test_split_percentage'] = args.test_split_percentage
    if args.seq_len_max is not None:
        configs['seq_len_max'] = args.seq_len_max

    if args.use_generated_data_efficacy is not None:
        configs['use_generated_data_efficacy'] = args.use_generated_data_efficacy
