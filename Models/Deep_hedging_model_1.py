import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

df_long = pd.read_parquet('Data/Data.parquet')

class HedgingDataset(Dataset):
    def __init__(self, df, seq_len=10):
        super().__init__()
        """
        df cols: 'n_sim', 'time', 'S', 'v'
                
        """
        self.df = df
        self.seq_len = seq_len
        self.sim_groups = df.groupby('n_sim')
        self.sim_indexes = df['n_sim'].unique()

    def __len__(self):
        # call len(df)
        # n of samples = n_sim * (steps - seq_len)
        return sum(len(group) - self.seq_len for _, group in self.sim_groups)
    
    def __getitem__(self, index):
        # to get the position like df[i] (i\in\mathbb(N) lol)
        # tensor is R^(seq_len*2) since I used S and v
        for sim_index in self.sim_indexes:
            group = self.sim_groups.get_group(sim_index)
            max_start = len(group) - self.seq_len
            if index < max_start:
                start = index
                end = start + self.seq_len
                seq = group.iloc[start:end][['S', 'v']].values
                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                return seq_tensor
            index -= max_start
        raise IndexError

















