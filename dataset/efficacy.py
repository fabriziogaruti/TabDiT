import os
from os import path
import pandas as pd
import tqdm
from loguru import logger
from torch.utils.data.dataset import Dataset


class EfficacyDataset(Dataset):

    def __init__(self,
                 root="./generated_data/pkdd99/merged_s0",
                 dataset_name_to_use="pkdd99",
                 seq_len_max=-1,
                 use_generated_data=True,):

        self.use_generated_data = use_generated_data
        self.root = root
        self.dataset_name_to_use = dataset_name_to_use

        self.seq_len_max = seq_len_max
        if self.seq_len_max < 0:
            self.seq_len_max = float('inf')

        self.trans_table = None
        self.parent_table = None

        self.dataset_dataframe = pd.DataFrame()
        self.parent_dataset_dataframe = pd.DataFrame()
        self.labels = []

        self.encode_data()
        self.user_level_data()


    def __len__(self):
        return len(self.labels)


    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')


    def user_level_data(self):
        for user_code, user_data in tqdm.tqdm(self.trans_table.groupby(by='user')):

            if self.use_generated_data:
                type_trans_string = "generated"
            else:
                type_trans_string = "real"
            user_data = user_data[user_data['type_trans_discriminator'] == type_trans_string]

            if not (type_trans_string, user_code) in self.parent_table.index:
                logger.error(f"user_parent empty, type_trans_string: {type_trans_string}, user: {user_code}")
                continue

            user_parent_data = self.parent_table.loc[[(type_trans_string, user_code)]]

            if len(user_data) > self.seq_len_max:
                user_data = user_data.head(self.seq_len_max)

            user_data = user_data.drop(['user','type_trans_discriminator'], axis=1)

            user_trans = user_data.to_numpy()
            if user_trans.shape[0] == 0:
                logger.error(f"user_trans empty, type_trans_string: {type_trans_string}, user: {user_code}")
                continue

            if self.dataset_name_to_use == "age2":
                self.labels += [float(user_parent_data['age'].iloc[0]) > 30]
            elif self.dataset_name_to_use == "pkdd99":
                self.labels += ['Moravia' in user_parent_data['region'].iloc[0]]
            elif self.dataset_name_to_use == "rossmann":
                self.labels += [user_parent_data['Promo2'].iloc[0]]
            elif self.dataset_name_to_use == "airbnb":
                self.labels += [int(user_parent_data['n_sessions'].iloc[0]) >= 20]
            elif self.dataset_name_to_use == "bbpm_categorizzato":
                valore_label = (user_parent_data['eta_range'].iloc[0])
                if valore_label != 'None':
                    self.labels += [float(valore_label) > 40]
                else:
                    self.labels += [True]
            else:
                logger.warning("Datatset not found for efficacy")
                self.labels += [user_parent_data['label'].iloc[0]]
            user_data['user_id'] = user_code
            self.dataset_dataframe = pd.concat([self.dataset_dataframe, user_data])


    def encode_data(self):
        data_file = path.join(self.root, f"true_trans.parquet")
        data_real = pd.read_parquet(data_file)
        logger.info(f"read data real: {data_real.shape}")
        data_file = path.join(self.root, f"generated_trans.parquet")
        data_generated = pd.read_parquet(data_file)
        logger.info(f"read data generated: {data_generated.shape}")

        data = pd.concat([data_real, data_generated], keys=['real', 'generated'])
        data = data.rename_axis(['type_trans_discriminator', 'index2']).reset_index()
        data.drop(columns=['index2'], inplace=True)
        logger.info(f"read data : {data.shape}")
        logger.info(f"{data_file} is read.")

        for col in data.columns:
            data[col] = self.nanNone(data[col])
        data.fillna("None", inplace=True)
        self.trans_table = data

        logger.info("reading parent true and pred")
        parent_real_file = path.join(self.root, f"true_parent.parquet")
        parent_real = pd.read_parquet(parent_real_file)

        parent_generated_file = path.join(self.root, f"generated_parent.parquet")
        if os.path.exists(parent_generated_file):
            parent_generated = pd.read_parquet(parent_generated_file)
            parent_data = pd.concat([parent_real, parent_generated], keys=['real', 'generated'])
        else:
            logger.warning(f'Only the true parent table is present, no prediction. Setting generated parent with true parent!')
            parent_data = pd.concat([parent_real, parent_real], keys=['real', 'generated'])

        parent_data = parent_data.rename_axis(['type_trans_discriminator', 'index2']).reset_index()
        parent_data.drop(columns=['index2'], inplace=True)

        parent_data.sort_values(by=['user', 'type_trans_discriminator'], inplace=True, ascending=True)
        parent_data.set_index(['type_trans_discriminator', 'user'], inplace=True)

        for col in parent_data.columns:
            parent_data[col] = self.nanNone(parent_data[col])
            parent_data[col] = parent_data[col].astype(str)
        parent_data.fillna("None", inplace=True)
        self.parent_table = parent_data
