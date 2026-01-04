import os
import torch
import safetensors.torch
from modules import shared, sd_models


def get_checkpoint_files():
    """Get list of available checkpoint files"""
    checkpoint_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
    if not os.path.exists(checkpoint_dir):
        return []
    
    files = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith(('.ckpt', '.safetensors')):
            files.append(f)
    return sorted(files)


def load_checkpoint(filepath):
    """Load a checkpoint file"""
    if filepath.endswith('.safetensors'):
        return safetensors.torch.load_file(filepath)
    else:
        return torch.load(filepath, map_location='cpu')


def save_checkpoint(state_dict, filepath, use_safetensors=True):
    """Save a checkpoint file"""
    if use_safetensors:
        safetensors.torch.save_file(state_dict, filepath)
    else:
        torch.save(state_dict, filepath)


def merge_checkpoints(model_a_path, model_b_path, output_path, alpha=0.5, merge_mode='weight_sum', 
                     use_safetensors=True, multiplier=1.0):
    print(f"Loading model A: {model_a_path}")
    model_a = load_checkpoint(model_a_path)
    
    print(f"Loading model B: {model_b_path}")
    model_b = load_checkpoint(model_b_path)
    
    merged = {}
    
    if merge_mode == 'weight_sum':
        # Simple weighted sum: (1-alpha) * A + alpha * B
        for key in model_a.keys():
            if key in model_b:
                merged[key] = (1 - alpha) * model_a[key] + alpha * model_b[key]
            else:
                merged[key] = model_a[key]
    
    elif merge_mode == 'add_difference':
        # Add difference: A + multiplier * alpha * (B - A)
        for key in model_a.keys():
            if key in model_b:
                diff = model_b[key] - model_a[key]
                merged[key] = model_a[key] + multiplier * alpha * diff
            else:
                merged[key] = model_a[key]
    
    else:
        # Default to weight_sum if unknown mode
        for key in model_a.keys():
            if key in model_b:
                merged[key] = (1 - alpha) * model_a[key] + alpha * model_b[key]
            else:
                merged[key] = model_a[key]
    
    print(f"Saving merged model to: {output_path}")
    save_checkpoint(merged, output_path, use_safetensors)
    return output_path

