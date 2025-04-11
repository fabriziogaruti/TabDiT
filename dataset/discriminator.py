import os
from os import path
import pandas as pd
import tqdm
from loguru import logger
from torch.utils.data.dataset import Dataset


class DiscriminatorDataset(Dataset):

    def __init__(self,
                 root="./generated_data/czech/merged_s0",
                 discriminate_with_parent=False,
                 seq_len_max=-1,):

        self.root = root
        self.discriminate_with_parent = discriminate_with_parent
        self.seq_len_max = seq_len_max
        if self.seq_len_max < 0:
            self.seq_len_max = float('inf')

        self.trans_table = None
        if self.discriminate_with_parent:
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
        for user_code, user_data in tqdm.tqdm(self.trans_table.groupby(by=['user', 'type_trans_discriminator'])):

            user_id = f'{user_code[0]}-{user_code[1]}'
            self.labels += [int(user_code[1] == 'generated')]

            if len(user_data) > self.seq_len_max:
                user_data = user_data.head(self.seq_len_max)

            user_data = user_data.drop(['user','type_trans_discriminator'], axis=1)

            user_data['user_id'] = user_id
            self.dataset_dataframe = pd.concat([self.dataset_dataframe, user_data])

            if self.discriminate_with_parent:
                user_parent_data = self.parent_table.loc[[(user_code[1], user_code[0])]]
                user_parent_data.reset_index(drop=True, inplace=True)

                user_parent_data['user_id'] = user_id
                self.parent_dataset_dataframe = pd.concat([self.parent_dataset_dataframe, user_parent_data])


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

        if self.discriminate_with_parent:
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
