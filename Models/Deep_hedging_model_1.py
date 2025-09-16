import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

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
def eur_call_payoff(S_seq, strike=100):
    S_T = S_seq[:, -1, 0]
    return torch.clamp(S_T - float(strike), min=0.0)

def eur_put_payoff(S_seq, strike=100):
    S_T = S_seq[:, -1, 0]
    return torch.clamp(float(strike) - S_T, min=0.0)

def lookback_call_payoff(S_seq, strike=100):
    S_max = S_seq.max(dim=1).values[:,0]  # max over time, return batch size
    return torch.clamp(S_max - float(strike), min=0.0)

def asian_call_payoff(S_seq, strike=100):
    S_avg = S_seq.mean(dim=1)[:,0]
    return torch.clamp(S_avg - float(strike), min=0.0)

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
    # both tensor and float for cost_rate
    if not torch.is_tensor(cost_rate):
        if float(cost_rate) == 0.0:
            return torch.zeros(delta.shape[0], device=delta.device, dtype=delta.dtype)
    else:
        if torch.all(cost_rate == 0):
            return torch.zeros(delta.shape[0], device=delta.device, dtype=delta.dtype)
   
    # previous positions (delta_{-1}=0)
    delta_prev = torch.zeros_like(delta)
    delta_prev[:, 1:, :] = delta[:, :-1, :]

    trades = delta - delta_prev               # [batch, T, d]
    abs_trades = trades.abs()

    # takes both tensor and float argument for costs, shaped tensor [,d]
    if torch.is_tensor(cost_rate):
        cr = cost_rate.view(1, 1, -1).to(delta.device).type_as(delta)
    else:
        cr = float(cost_rate)

    # cost per element and sum
    costs = (abs_trades * S_seq_at_trade * cr).sum(dim=(1, 2))  # [batch], sum over dim 1 and 2 (T and d)
    return costs

def comp_PL_T_of_delta(delta, S_seq, payoff_fn=None, payoff_kwargs=None, cost_rate=0.0, p0=0.0):
    batch = S_seq.shape[0]
    device = S_seq.device
    dtype = S_seq.dtype

    if payoff_fn is None:
        Z = torch.zeros(batch, device=device, dtype=dtype)
    else:
        Z = payoff_fn(S_seq, **(payoff_kwargs or {})) # use full path
        Z = Z.type_as(S_seq)

    delta_dot_s = comp_delta_dot_S(delta, S_seq)
    costs = comp_transaction_costs(delta, S_seq[:, :-1, :], cost_rate=cost_rate)
    pl = -Z + float(p0) + delta_dot_s - costs
    return pl


"""
Hedging Model (Semi-recurrent hedging model)
Following setup Buhler & Teichmann
(Using only S_k and delta_{k-1} as features, in further models I'll use also v_k,
that's why n_assets*2 and not *3)

"""
class HedgingNeuralNetwork(nn.Module):
    def __init__(self, n_assets, layer_size=16, n_layers=2):
        super().__init__()
        self.n_assets = n_assets

        input_dim = n_assets * 2
        # nn.Linear(input_size, output_size)
        layers = [nn.Linear(input_dim, layer_size), nn.ReLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(layer_size, layer_size), nn.ReLU()]
        layers.append(nn.Linear(layer_size, n_assets)) # output n_asset which is d in the paper
        
        self.net = nn.Sequential(*layers)

    def forward(self, S_seq):
        B, Tplus1, d = S_seq.shape
        T = Tplus1 - 1
        assert d == self.n_assets

        delta_seq = []
        delta_prev = torch.zeros(B, d, device=S_seq.device, dtype=S_seq.dtype)

        for k in range(T):
            S_k = S_seq[:,k,:] # [B,d]
            input = torch.cat([S_k, delta_prev], dim=-1) # [B,2d]
            delta_k = self.net(input) # [B,d]
            delta_seq.append(delta_k)
            delta_prev = delta_k

        delta = torch.stack(delta_seq, dim=1)   # [B,T,d]
        return delta


class EntropicLoss(nn.Module):
    """
    rho_ent(pl) = (1/lambda) * log E[ exp(-lambda * pl) ]
    Minimize rho_ent(pl). Lower is better.
    """
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = float(lam)

    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        # pl: [batch]
        x = -self.lam * pl
        m = torch.max(x)       # makes computation safe otherwise kaboom (ensures training don't blow up)
        return (torch.log(torch.mean(torch.exp(x - m))) + m) / self.lam











# To see if everything works

#device = torch.device("cpu")

# get only one batch (basically I break the cycle)
for S, v in train_loader:
    # S: list-> because random_split returns Subset; collate should give tensors if same lengths
    # but with DataLoader over Subset we get (batch, T+1, 1). If not, adapt accordingly.
    # Move to device and ensure dtype
    S = S.to(device)
    v = v.to(device)
    break

# test_costs.py (run in same notebook where train_loader exists)

B, Tplus1, d = S.shape
T = Tplus1 - 1
d = d  # usually 1

# dummy delta: zeros (no hedging)
delta = torch.zeros(B, T, d, dtype=S.dtype, device=device)


# compute payoff Z as European call strike=100
Z = eur_call_payoff(S, strike=100.0)  # [batch]

pl = comp_PL_T_of_delta(delta, S, payoff_fn=eur_call_payoff, payoff_kwargs={'strike':100.0}, cost_rate=0.0)
print("Shapes -> S:", S.shape, "delta:", delta.shape, "Z:", Z.shape, "PL:", pl.shape)
print("PL sample:", pl)


# basically should retun [batch] with values -Z (considring delta = 0 for delta dim [B,T,d])

print(f"Shapes: batch={B}, T+1={Tplus1}, d={d}")

# init model
layer_size = d+15
model = HedgingNeuralNetwork(n_assets=d, layer_size=layer_size, n_layers=2).to(device)

# forward pass
delta = model(S)
print("Output delta shape:", delta.shape)  # expect [B, T, d]



    














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



















