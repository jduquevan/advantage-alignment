def rank_dict(dictionary):
    ranked_dict = {
        k: v
        for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)
    }
    return ranked_dict


from collections import Counter, defaultdict
import numpy as np
import torch
from copy import deepcopy
import operator


def is_scalar(d):
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in d.values()):
        return True
    elif all(isinstance(v, (np.ndarray, list, torch.Tensor)) for v in d.values()):
        return False
    else:
        raise ValueError(
            "The values of the dictionary must be either all scalars or all vectors."
        )


def sum_dicts(dict_list):
    if is_scalar(dict_list[0]):
        return dict(sum((Counter(dict_) for dict_ in dict_list), Counter()))
    else:
        dict_list = [{k: np.array(v) for k, v in d.items()} for d in dict_list]
        first_array_shape = next(iter(dict_list[0].values())).shape
        sum_dict = defaultdict(lambda: np.zeros(first_array_shape))
        list(
            map(
                lambda d: list(
                    map(lambda kv: (sum_dict[kv[0]].__iadd__(kv[1])), d.items())
                ),
                dict_list,
            )
        )

        return dict(sum_dict)


def empty_lists_in_nested_dict(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dict[key] = empty_lists_in_nested_dict(value)
        elif isinstance(value, list):
            new_dict[key] = []
        else:
            new_dict[key] = value
    return new_dict


def a2t(a, bs=1, device=None):
    # Convert a numpy array to a batchfy pytorch tensor
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(a, torch.Tensor):
        if bs == 1 and a.shape[0] != 1:
            return a.unsqueeze(0).to(device)
        else:
            assert a.shape[0] == bs, "batch size does not match"
            return a.to(device)
    else:
        raise ValueError(
            f"a is neither a numpy array nor a torch tensor, it is {type(a)}"
        )


def t2a(t):
    # Convert a batchfy pytorch tensor to a numpy array
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().reshape(-1)
    elif isinstance(t, np.ndarray):
        return t.reshape(-1)
    else:
        raise ValueError(f"t is not a torch tensor, it is {type(t)}")


def get_property_from_vec_env(vectorize_propoerty):
    if isinstance(vectorize_propoerty, (int, float, bool)):
        output = vectorize_propoerty
    else:
        if isinstance(vectorize_propoerty, torch.Tensor):
            assert (
                torch.unique(vectorize_propoerty).size(0) == 1
            ), "vectorize_propoerty should be the same for all agents"
            output = vectorize_propoerty[0].item()
        elif isinstance(vectorize_propoerty, np.ndarray):
            assert (
                np.unique(vectorize_propoerty).size == 1
            ), "vectorize_propoerty should be the same for all agents"
            output = vectorize_propoerty[0]
    return output


def batchfy_lt(list_of_tensor, bs, device=None):
    # convert a list of tensors to a batchfy tensor
    # input: [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])], bs = 3
    # output: tensor([[1, 2], [3, 4], [5, 6]])
    # input torch.tensor([1, 2]) # bs = 1
    # output torch.tensor([[1, 2]])
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if isinstance(list_of_tensor, (list, tuple)):
        return torch.stack(list_of_tensor).to(device)
    elif isinstance(list_of_tensor, (np.ndarray)):
        return torch.stack(list_of_tensor.tolist()).to(device)
    elif isinstance(list_of_tensor, torch.Tensor) and list_of_tensor.shape[0] != bs:
        return torch.stack([list_of_tensor]).to(device)
    elif isinstance(list_of_tensor, torch.Tensor) and list_of_tensor.shape[0] == bs:
        return list_of_tensor.to(device)
    else:
        raise ValueError(
            f"list_of_tensor is neither a list nor a torch tensor, it is {type(list_of_tensor)}"
        )


def nested_dict_to(nested_dict, device=None):
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    nested_dict_ = deepcopy(nested_dict)
    for key, value in nested_dict_.items():
        if isinstance(value, dict):
            nested_dict_to(value)
        elif isinstance(value, torch.Tensor):
            nested_dict_[key] = value.to(device)
    return nested_dict_


def a2t_in_nested_dict(nested_dict, device=None):
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    nested_dict_ = deepcopy(nested_dict)
    for key, value in nested_dict_.items():
        if isinstance(value, dict):
            a2t_in_nested_dict(value)
        else:
            nested_dict_[key] = torch.tensor(value).to(device)
    return nested_dict_


numpy_ops = {
    "==": operator.eq,
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
}
torch_ops = {
    "==": torch.eq,
    "<=": torch.le,
    ">=": torch.ge,
    "<": torch.lt,
    ">": torch.gt,
}

def compare(x, y, op_str, allany=None):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        op = numpy_ops[op_str]
        if allany is None:
            return op(x, y)
        elif allany == "all":
            return np.all(op(x, y))
        elif allany == "any":
            return np.any(op(x, y))
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        op = torch_ops[op_str]
        if allany is None:
            return op(x, y)
        elif allany == "all":
            return torch.all(op(x, y))
        elif allany == "any":
            return torch.any(op(x, y))
    else:
        raise TypeError("Inputs must be both NumPy arrays or both PyTorch tensors")

def compare_eps(x, y, eps=0.1):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        abs_diff = np.abs(x - y)
        return np.all(abs_diff <= eps)
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        abs_diff = torch.abs(x - y)
        return torch.all(abs_diff <= eps)
    else:
        raise TypeError("Inputs must be both NumPy arrays or both PyTorch tensors")
