import numpy as np
import yaml
import torch


def parse_yaml(yaml_config):
    
    if yaml_config is not None:
        with open(yaml_config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    # default settings
    else:
        config = {}
        config["paths"] = {
            "root": "../",
            "save_dir": "../training_outputs",
            "model_name": "graph_decoder"
        }
        config["model_settings"] = {
            "hidden_channels_GCN": [32, 128, 256, 512, 512, 256, 256],
            "hidden_channels_MLP": [256, 128, 64],
            "num_classes": 12
        }
        config["graph_settings"] = {
            "code_size": 7,
            "error_rate": 0.001,
            "m_nearest_nodes": 10,
            "power": 2
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config["training_settings"] = {
            "seed": None,
            "dataset_size": 50000,
            "batch_size": 4096,
            "epochs": 5,
            "lr": 0.01,
            "device": device,
            "resume_training": False,
            "current_epoch": 0
        }
    
    # read settings into variables
    paths = config["paths"]
    model_settings = config["model_settings"]
    graph_settings = config["graph_settings"]
    training_settings = config["training_settings"]
    
    return paths, model_settings, graph_settings, training_settings
