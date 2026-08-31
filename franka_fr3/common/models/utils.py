import torch
from .cosdf import COSDF

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    data = {"id": model.id, "meta": model.meta, "state_dict": model.state_dict()}
    torch.save(data, path)


def load_model(path, device: str):
    data = torch.load(path, map_location=device)
    if data["id"] != "COSDF":
        raise ValueError(f"Model type {data['id']} not recognized")
    n_layers, n_hidden = data["meta"][:2]
    model = COSDF(n_layers, n_hidden, device=device)
    model.load_state_dict(data["state_dict"])
    return model.to(device)
