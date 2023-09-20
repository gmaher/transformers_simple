import torch
import numpy as np
def generate_sample(x, model, length):
    """
        x - list of sequence ids
    """

    l = len(x)

    seq_arr = [0]*(l+length)
    seq_arr[:l] = x

    for t in range(l,l+length):
        input = seq_arr[t-l:t]

        input = torch.tensor(input).view(1,-1)

        o = model(input)[0]

        p = torch.nn.functional.softmax(o, dim=0).data.numpy()

        next_id = np.random.choice(len(p), p=p)

        seq_arr[t] = next_id

    return seq_arr[l:]
