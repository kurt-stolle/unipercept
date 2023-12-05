#%%

import h5py

ds = h5py.File("test.h5", "w")
ds.


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

    data[2:3] = data[2:3].apply(lambda x: x * 2)
    data.device = "cuda"

#%%


import torch
import tempfile

from tensordict import PersistentTensorDict, TensorDict 

with tempfile.NamedTemporaryFile("w") as f:
    print(f.name)

    data = PersistentTensorDict(batch_size=[3], filename=f.name, mode="w")
    data[2] = {
        "abc": torch.randn(1, 1),
        "def": torch.randn(1, 2)
    }

    print(data)
    print(data["abc"])
    print(data["def"])