#%%

import h5py

#%%

import torch
import tempfile

from tensordict import PersistentTensorDict, TensorDict 

with tempfile.NamedTemporaryFile("w") as f:
    print(f.name)

    data = PersistentTensorDict(batch_size=[3], filename=f.name, mode="w")
    data["abc"] = torch.randn(3, 100, 100)

    data["a", "b"] = torch.randn(3,3)

    print(data[2])