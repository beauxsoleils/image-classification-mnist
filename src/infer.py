
import torch 

def run_inference(model, input_tensor):
    model.eval()
    with model.no_grad():
        output = model(input_tensor)