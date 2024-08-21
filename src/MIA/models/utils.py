import torch
import os
import dataclasses

def save_torch_model(model, save_location):
    # torch.jit.save(model, save_location)
    state_dict = model.state_dict()
    args = dataclasses.asdict(model.args)
    state_dict['model_args'] = args
    torch.save(state_dict, save_location)

def load_torch_model(ModelClass, model_path, enforce_load=True):
    if model_path != 'none' and os.path.exists(model_path):
        if not os.path.isfile(model_path):
            raise IOError(f'Path provided to load_torch_model ({model_path}) is not a file.')
        state_dict = None
        device='cuda'
        if not torch.cuda.is_available():
            state_dict = torch.load(model_path, map_location=torch.device('cpu')) # If only CPU available then need to ensure weights are mapped to CPU if it was saved from GPU model
            device = 'cpu'
        else:
            state_dict = torch.load(model_path) # If a GPU is available weight loading doesn't matter
        model_arguments = state_dict['model_args']
        del state_dict['model_args']
        model = ModelClass(**model_arguments).to(device)
        model.load_state_dict(state_dict)
        print(f'Succesfully loaded model {model_path}')
        return model

    if enforce_load:
        raise IOError(f'Failed to load existing model: {model_path}')
    print('Warning - loading model failed. Okay if creating initial model.') #TODO: Don't print this if existing_model_path == None
    return None
