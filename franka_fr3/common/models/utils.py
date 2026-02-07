import torch
from .sll import DenseSDP
from .gndf import G_NDF

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
    elif model_type in ["G-NDF", "G-NDF-Sparse", "G-NDF-Refined", "G-NDF-7D-Refined", "G-NDF-7D"]:
        # New format: [n_layers, n_hidden, n_points_per_link]
        # Legacy format: [n_layers, n_hidden] or with frequencies
        meta = data["meta"]
        if len(meta) >= 3:
            n_layers, n_hidden, n_points = meta[:3]
            model = G_NDF(n_layers, n_hidden, n_points, device=device)
        else:
            # Legacy: [n_layers, n_hidden]
            model = G_NDF(meta[0], meta[1], 2, device=device)
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
            return G_NDF(
                n_layers=n_layers, 
                n_hidden=n_hidden, 
                n_points_per_link=kwargs.get("n_points_per_link", 2),
                device=kwargs.get("device", "cuda")
            )

        case _:
            raise Exception(f"model name {name} not recognized")
