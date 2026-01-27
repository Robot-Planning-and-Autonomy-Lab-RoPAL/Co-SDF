import torch
from .sll import DenseSDP
import torch.nn as nn

class PeriodicEmbedding(nn.Module):
    def __init__(self, frequencies=[1, 2, 4, 8]):
        super().__init__()
        self.frequencies = frequencies
    def forward(self, x):
        # x: (B, DIM)
        embeddings = []
        for f in self.frequencies:
            embeddings.append(torch.cos(f * x))
            embeddings.append(torch.sin(f * x))
        return torch.cat(embeddings, dim=-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    data = { "id": model.id, "meta" : model.meta, "state_dict" : model.state_dict()}
    torch.save(data, path)
    
def load_model(path, device:str):
    data = torch.load(path, map_location=device)
    model_type = data.get("id","Spectral")
    if model_type == "SDP":
        model = DenseSDP(*data["meta"])
    elif model_type == "G-NDF-2D":
        from .gndf_2d import G_NDF_2D
        # meta format: [n_layers, n_hidden, n_points_per_link, frequencies (unused), base_position, link_lengths]
        n_layers, n_hidden, n_points_per_link, _freqs_unused, base_position, link_lengths = data["meta"]
        model = G_NDF_2D(
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_points_per_link=n_points_per_link,
            base_position=base_position,
            link_lengths=link_lengths,
            device=device,
        )
    else:
        raise Exception(f"Model type {model_type} not recognized")
    model.load_state_dict(data["state_dict"])
    return model.to(device)

def select_model(name, DIM, n_layers, n_hidden, **kwargs):
    match name.lower():
        case "mlp":
            final_activ = kwargs.get("final_activ", torch.nn.Identity)
            return MultiLayerPerceptron(
                DIM, n_hidden, n_layers, 
                final_activ=final_activ)
        
        case "siren":
            return SirenNet(DIM, n_hidden, n_layers)

        case "ortho":
            return DenseLipNetwork(
                DIM, n_hidden, n_layers,
                group_sort_size = kwargs.get("group_sort_size",0), 
                niter_spectral = kwargs.get("niter_spectral", 3), 
                niter_bjorck = kwargs.get("niter_bjorck", 15))

        case "sll":
            return DenseSDP(DIM, n_hidden, n_layers)

        case "phase":
            return PhaseNet(DIM, n_hidden, n_layers, FF=kwargs.get("FF", False), skip_in=(4,))

        case "gndf":
            from .gndf_2d import G_NDF_2D
            return G_NDF_2D(
                n_layers, 
                n_hidden, 
                n_points_per_link=kwargs.get("n_points_per_link", 2),
                base_position=kwargs.get("base_position", [5.0, 5.0]),
                link_lengths=kwargs.get("link_lengths", [2.1, 2.1]),
                device=kwargs.get("device", "cuda")
            )
        
        case _:
            raise Exception(f"model name {name} not recognized")
