import numpy as np
import torch
import os
import time
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from src.GNN_Decoder import GNN_Decoder 
import src.gnn_models as gnn
import stim

if __name__ == '__main__':
    # Code and noise settings
    code_size = int(os.getenv('CODE_SIZE'))
    training_error_rate = [0.001, 0.002, 0.003, 0.005]
    repetitions = int(os.getenv('REPETITIONS')) if os.getenv('REPETITIONS') is not None else code_size

    # Training settings
    num_iterations = int(os.getenv('NUM_ITERATIONS')) if os.getenv('NUM_ITERATIONS') is not None else None
    batch_size = int(os.getenv('BATCH_SIZE')) if os.getenv('BATCH_SIZE') is not None else None

    learning_rate = float(os.getenv('LEARNING_RATE')) if os.getenv('LEARNING_RATE') is not None else None
    criterion = torch.nn.BCEWithLogitsLoss()
    manual_seed = int(os.getenv('MANUAL_SEED')) if os.getenv('MANUAL_SEED') is not None else None
    benchmark = bool(os.getenv('BENCHMARK')) if os.getenv('BENCHMARK') is not None else False
    buffer_size = int(os.getenv('BUFFER_SIZE')) if os.getenv('BUFFER_SIZE') is not None else None
    replacements_per_iteration = int(os.getenv('REPLACEMENTS_PER_ITERATION')) if os.getenv('REPLACEMENTS_PER_ITERATION') is not None else None
    test_size = int(os.getenv('TEST_SIZE')) if os.getenv('TEST_SIZE') is not None else None

    # Graph settings
    num_node_features = int(os.getenv('NUM_NODE_FEATURES')) if os.getenv('NUM_NODE_FEATURES') is not None else None
    power = float(os.getenv('EDGE_WEIGHT_POWER')) if os.getenv('EDGE_WEIGHT_POWER') is not None else None
    m_nearest_nodes = int(os.getenv('M_NEAREST_NODES')) if os.getenv('M_NEAREST_NODES') is not None else None
    cuda = bool(os.getenv('USE_CUDA')) if os.getenv('USE_CUDA') is not None else True
    validation = bool(os.getenv('USE_VALIDATION')) if os.getenv('USE_VALIDATION') is not None else False

    # IO settings
    job_id = os.getenv('SLURM_JOB_ID') 
    if os.getenv('SLURM_ARRAY_JOB_ID') is None:
        job_id = os.getenv('SLURM_JOB_ID') # If not array job, use job ID
    else:
        job_id = os.getenv('SLURM_ARRAY_JOB_ID') # If array job, use array job ID
    if os.getenv('SLURM_ARRAY_TASK_ID') is None:
        array_id = None
    else:
        array_id = os.getenv('SLURM_ARRAY_TASK_ID') # If array job, get array task ID
    job_name = os.getenv('JOB_NAME')
    save_directory_path = os.getenv('SLURM_SUBMIT_DIR') # Save to login node disk
    save_directory_path = os.path.join(save_directory_path, 'results/') # Add node local results dir to path
    # If running array jobs, adjust filename to include both array and job id
    if array_id is None:
        filename_prefix = f'{job_name}_id{job_id}' 
    else:
        filename_prefix = f'{job_name}_id{job_id}_{array_id}' 
    # If specified, resume run by loading Decoder attributes from file (history and model/optim state dicts)
    resumed_training_file_name = str(os.getenv('RESUMED_TRAINING_FILE_NAME')) if os.getenv('RESUMED_TRAINING_FILE_NAME') is not None else None

    GNN_params = {
        'model': {
            'class': gnn.GNN_7,
            'num_classes': 1, # 1 output class for two-headed model
            'loss': criterion,
            'num_node_features': num_node_features,
            'initial_learning_rate': learning_rate,
            'manual_seed': manual_seed
        },
        'graph': {
            'num_node_features': num_node_features,
            'm_nearest_nodes': m_nearest_nodes,
            'power': power
        },
        'cuda': cuda,
        'save_path': save_directory_path, 
        'save_prefix': filename_prefix
    }

    # INITIALIZE DECODER, SET PARAMETERS
    print('\n==== DECODER PARAMETERS ====')
    decoder = GNN_Decoder(GNN_params)
    print(decoder.params)
    print(f'Code size: {code_size}\n')
    print(f'Repetitions: {repetitions}\n')
    print(f'Training error rate: {training_error_rate}\n')

    # LOAD MODEL AND TRAINING HISTORY FROM FILE
    # If specified, continue run by loading Decoder attributes from file (history and model/optim state dicts)
    if resumed_training_file_name is not None:
        load_path = os.getenv('SLURM_SUBMIT_DIR')
        load_path = os.path.join(load_path, 'results/', resumed_training_file_name + '.pt')
        print(('\nLoading training history, weights and optimizer to resume training '
            f'from {load_path}'))
        device = torch.device('cuda')
        current_device_id = torch.cuda.current_device()
        loaded_attributes = torch.load(load_path, map_location=f'cuda:{current_device_id}')
        decoder.load_training_history(loaded_attributes)
        decoder.model.to(device)
        
    # TRAIN
    print('\n==== TRAINING ====')
    decoder.train_with_data_buffer(
        code_size = code_size,
        repetitions = repetitions,
        error_rate = training_error_rate,
        train = True,
        save_to_file = True,
        batch_size = batch_size,
        learning_rate = learning_rate,
        num_iterations = num_iterations,
        benchmark = benchmark,
        buffer_size = buffer_size,
        replacements_per_iteration = replacements_per_iteration,
        test_size = test_size,
        learning_scheduler = False,
        validation = validation)

    print('\n==== TESTING ====')
    rates = [0.001, 0.002, 0.003]
    for r in rates:
        acc = decoder.train_with_data_buffer(
            code_size = code_size,
            repetitions = repetitions,
            error_rate = r,
            train = False,
            wandb_log = False,
            test_size = test_size)
        print(f'Test accuracy: {acc}, Error rate: {r}')