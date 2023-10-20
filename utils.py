import torch

def get_available_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def get_model_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None and p.grad.data is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def weights_biases_sum(model):
    total_weight_sum = 0.0
    for param in model.parameters():
        total_weight_sum += param.data.sum().item()
    return total_weight_sum