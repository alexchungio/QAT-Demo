import os
import time
import torch
import numpy as np
import random


def time_cost(f):
    def time_decorator(*args, **kwargs):

        start = time.perf_counter()
        out = f(*args, **kwargs)
        end = time.perf_counter()
        print(f'time cost {end - start}')

        return out

    return time_decorator


def set_random_seeds(random_seed=2022):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, save_path, full_model=False):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if full_model:
        torch.save(model, save_path)
    else:
        torch.save(model.state_dict(), save_path)


def load_model(ckpt_path, model=None, device=None, strict=True, full_model=False):

    if full_model:
        model = torch.load(ckpt_path, map_location=device)
    else:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=strict)

    return model


def load_torchscript(model_path, device=None):

    model = torch.jit.load(model_path, map_location=device)

    return model


def save_torchscript_model(model, model_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.jit.save(torch.jit.script(model), model_path)


def check_fuse(model, model_fuse, input_size=(10, 3, 32, 32), rtol=1e-4, atol=1e-6):
    model.eval()
    model_fuse.eval()
    input_data = torch.rand(input_size)
    out = model(input_data)
    out_fuse = model_fuse(input_data)
    assert np.allclose(out.detach().numpy(),
                       out_fuse.detach().numpy(),
                       rtol=rtol, atol=atol), print("Model equivalence test failed")
