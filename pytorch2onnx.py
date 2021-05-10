from net.st_gcn import Model

import torch
import torch.nn as nn
import torch.onnx
import numpy as np

weights_path = 'models/epoch50_model.pt'
onnx_model = "stgcn.onnx"

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    
    device = torch.device("cpu")

    model = Model(in_channels=3, num_class=400, edge_importance_weighting=True, graph_args={'layout': 'openpose', 'strategy': 'spatial'})

    state_dict = torch.load(weights_path)

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    input_data = np.random.randn(1, 36, 3, 300).astype(np.float32)

    dummy_input = torch.from_numpy(input_data).float().to(device)   

    torch.onnx.export(model,               # model being run
                (dummy_input), #, dummy_A),  # model input (or a tuple for multiple inputs)
                onnx_model,              #where to save the model (can be a file or file-like object)
                verbose=True,
                opset_version=11,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                )


        print('model generated')
