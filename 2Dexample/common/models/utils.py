import torch
from .sll import DenseSDP
from .cosdf import COSDF

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    data = {"id": model.id, "meta": model.meta, "state_dict": model.state_dict()}
    torch.save(data, path)


def load_model(path, device: str):
    data = torch.load(path, map_location=device)
    model_type = data["id"]
    if model_type == "SDP":
        model = DenseSDP(*data["meta"])
    elif model_type == "COSDF":
        n_layers, n_hidden, n_points_per_link, base_position, link_lengths = data["meta"]
        model = COSDF(
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_points_per_link=n_points_per_link,
            base_position=base_position,
            link_lengths=link_lengths,
            device=device,
        )
    else:
        raise ValueError(f"Model type {model_type} not recognized")
    model.load_state_dict(data["state_dict"])
    return model.to(device)


def select_model(name, DIM, n_layers, n_hidden, **kwargs):
    match name.lower():
        case "sll":
            return DenseSDP(DIM, n_hidden, n_layers)

        case "cosdf":
            return COSDF(
                n_layers,
                n_hidden,
                n_points_per_link=kwargs.get("n_points_per_link", 2),
                base_position=kwargs.get("base_position", [5.0, 5.0]),
                link_lengths=kwargs.get("link_lengths", [2.1, 2.1]),
                device=kwargs.get("device", "cuda"),
            )

        case _:
            raise ValueError(f"Model name {name} not recognized")
