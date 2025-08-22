import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split


"""
Dataset Setup

"""


# Dataset setup for Pytorch nn model
df_long = pd.read_parquet('Data/Data.parquet')

class HedgingDataset(Dataset):
    def __init__(self, df):
        """
        df cols: 'n_sim', 'time', 'S', 'v'
        Each sample = full path for one simulation.
        """
        super().__init__()
        self.paths = []
        for _, g in df.groupby("n_sim"):
            g_sorted = g.sort_values("time")
            S = torch.tensor(g_sorted['S'].values, dtype=torch.float32).unsqueeze(-1)
            v = torch.tensor(g_sorted['v'].values, dtype=torch.float32).unsqueeze(-1)
            self.paths.append((S, v))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]

df = HedgingDataset(df_long)
#try to add max -1 cpu cell usage
loader = DataLoader(df, batch_size=len(df), shuffle=True)

train_size = int(len(df)*0.6)
val_size = int(len(df)*0.2)
test_size = int(len(df)*0.2)

df_train, df_val, df_test = random_split(df, [train_size, val_size, test_size])

train_loader = DataLoader(df_train, batch_size=10, shuffle=False)
val_loader = DataLoader(df_val, batch_size=10, shuffle=False)
test_loader = DataLoader(df_test, batch_size=10, shuffle=False)



"""
Def for loss and objective functions

"""

def eur_call_payoff(S_T, strike=100):
    return torch.clamp(S_T - float(strike), min=0.0)

def comp_delta_dot_S(delta, S_seq):
    assert S_seq.shape[1] == delta.shape[1] + 1, "S_seq one more timestep than delta"
    price_diff = S_seq[:, 1:, :] - S_seq[:, :-1, :]   # [batch, T, d], difference along T axis
    hedge_gains = (delta * price_diff).sum(dim=(1, 2))  # sum over time and instruments
    return hedge_gains  # return [batch]

def comp_transaction_costs(delta, S_seq_at_trade, cost_rate=0.0):
    """
    Proportional costs
    returns: tensor [batch] total cost per path
    """
    if float(cost_rate) == 0.0:
        return torch.zeros(delta.shape[0], device=delta.device, dtype=delta.dtype)

    # previous positions (delta_{-1}=0)
    delta_prev = torch.zeros_like(delta)
    delta_prev[:, 1:, :] = delta[:, :-1, :]

    trades = delta - delta_prev    # [batch, T, d]
    abs_trades = trades.abs()

    # ensure cost_rate is a tensor shaped (d,)
    if torch.is_tensor(cost_rate):
        cr = cost_rate.view(1, 1, -1).to(delta.device).type_as(delta)
    else:
        cr = float(cost_rate)

    # cost per element and sum
    costs = (abs_trades * S_seq_at_trade * cr).sum(dim=(1, 2))  # [batch]
    return costs # retruns [batch]










"""
This idea is to create subsamples to avoid gradient explosion
Like 10 trading days for sample instead of the full lenght of the Heston path process generator

"""

# class HedgingDataset(Dataset):
#     def __init__(self, df, seq_len=10):
#         super().__init__()
#         """
#         df cols: 'n_sim', 'time', 'S', 'v'
                
#         """
#         self.df = df
#         self.seq_len = seq_len
#         self.sim_groups = df.groupby('n_sim')
#         self.sim_indexes = df['n_sim'].unique()

#     def __len__(self):
#         # call len(df)
#         # n of samples = n_sim * (steps - seq_len)
#         return sum(len(group) - self.seq_len for _, group in self.sim_groups)
    
#     def __getitem__(self, index):
#         # to get the position like df[i] (i\in\mathbb(N) lol)
#         # tensor is R^(seq_len*2) since I used S and v
#         for sim_index in self.sim_indexes:
#             group = self.sim_groups.get_group(sim_index)
#             max_start = len(group) - self.seq_len
#             if index < max_start:
#                 start = index
#                 end = start + self.seq_len
#                 S = torch.tensor(group.iloc[start:end]['S'].values, dtype=torch.float32).unsqueeze(-1)
#                 v = torch.tensor(group.iloc[start:end]['v'].values, dtype=torch.float32).unsqueeze(-1)
#                 return S, v
#             index -= max_start
#         raise IndexError


# df = HedgingDataset(df_long)

# # try to add max -1 cpu cell usage
# loader = DataLoader(df, batch_size=10, shuffle=False, num_workers=2, pin_memory=True)



















