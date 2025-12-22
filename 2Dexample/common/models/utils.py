import torch
from .sll import DenseSDP

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

        case _:
            raise Exception(f"model name {name} not recognized")
