import torch
import numpy as np
import torch.nn
import stim
import os

from src.utils import parse_yaml
from src.gnn_models import GNN_7
from src.graph_representation import sample_syndromes
from torch_geometric.nn import knn_graph

import wandb
os.environ["WANDB_SILENT"] = "True"

from pathlib import Path
from datetime import datetime
import copy
import time

class Decoder:
    def __init__(self, yaml_config=None, script_name=None):
        # load settings and initialise state
        paths, model_settings, graph_settings, training_settings = parse_yaml(yaml_config)
        self.save_dir = Path(paths["save_dir"])
        self.save_model_dir = Path(paths["save_model_dir"])
        self.saved_model_path = paths["saved_model_path"]
        self.model_settings = model_settings
        self.graph_settings = graph_settings
        self.training_settings = training_settings

        self.wandb_log = self.training_settings["wandb"]

        self.code_size = self.graph_settings["code_size"]
        self.d_t = self.graph_settings["d_t"]
        self.train_error_rate = self.graph_settings["train_error_rate"]
        self.test_error_rate = self.graph_settings["test_error_rate"]
        self.m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        self.power = self.graph_settings["power"]
        self.sigmoid = torch.nn.Sigmoid()

        # current training status
        self.epoch = training_settings["current_epoch"]
        if training_settings["device"] == "cuda":
            self.device = torch.device(
                training_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

       # create a dictionary saving training metrics
        training_history = {}
        training_history["epoch"] = self.epoch
        training_history["train_loss"] = []
        training_history["train_accuracy"] = []
        training_history["test_loss"] = []
        training_history["test_accuracy"] = []
        training_history["val_loss"] = []
        training_history["val_accuracy"] = []

        self.training_history = training_history
        
        # only keep best found weights
        self.optimal_weights = None

        # instantiate model and optimizer
        self.model = GNN_7(
            hidden_channels_GCN=model_settings["hidden_channels_GCN"],
            hidden_channels_MLP=model_settings["hidden_channels_MLP"],
            num_classes=model_settings["num_classes"]
            ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters())

        print(f'Running with a learning rate of {training_settings["lr"]}.')
        current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")

        # generate a unique name to not overwrite other models
        name = ("d" +
                str(graph_settings["code_size"]) +
                "_d_t_" +
                str(graph_settings["d_t"]) +
                '_' + current_datetime +
                '_' + script_name)
        save_path = self.save_dir / (name + ".pt")
        self.save_name = name

        # make sure we did not create an existing name
        if save_path.is_file():
            save_path = self.save_dir / (name + "_1.pt")

        save_model_path = self.save_model_dir / (name + "_model.pt")

        self.save_model_path = save_model_path
        self.save_path = save_path

        # check if model should be loaded
        if training_settings["resume_training"]:
            self.load_trained_model()

    def save_model_w_training_settings(self):
        # make sure the save folder exists, else create it
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_dir.mkdir(parents=True, exist_ok=True)

        attributes = {
            "training_history": self.training_history,
            "graph_settings": self.graph_settings,
            "training_settings": self.training_settings,
            "model_settings": self.model_settings,
        }

        attributes_model = {
            "training_history": self.training_history,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "graph_settings": self.graph_settings,
            "training_settings": self.training_settings,
            "model_settings": self.model_settings,
        }

        torch.save(attributes, self.save_path)
        torch.save(attributes_model, self.save_model_path)

    def load_trained_model(self):
        model_path = Path(self.saved_model_path)
        saved_attributes = torch.load(model_path, map_location=self.device, weights_only=False)

        # update attributes and load model with trained weights
        self.training_history = saved_attributes["training_history"]

        self.epoch = saved_attributes["training_history"]["epoch"] + 1
        self.model.load_state_dict(saved_attributes["model"])
        self.optimizer.load_state_dict(saved_attributes["optimizer"])

    def initialise_simulations(self, error_rate):
        # build the circuit, standard: d_t = code distance
        if error_rate.__class__ == float:
            circuit = stim.Circuit.generated(
                            "surface_code:rotated_memory_z",
                            rounds = self.d_t,
                            distance = self.code_size,
                            after_clifford_depolarization = error_rate,
                            after_reset_flip_probability = error_rate,
                            before_measure_flip_probability = error_rate,
                            before_round_data_depolarization = error_rate)
            self.compiled_sampler = circuit.compile_detector_sampler()
        
        elif error_rate.__class__ == list:
            compiled_samplers = []
            for p in error_rate:
                circuit = stim.Circuit.generated(
                            "surface_code:rotated_memory_z",
                            rounds = self.d_t,
                            distance = self.code_size,
                            after_clifford_depolarization = p,
                            after_reset_flip_probability = p,
                            before_measure_flip_probability = p,
                            before_round_data_depolarization = p)
                sampler = circuit.compile_detector_sampler()
                compiled_samplers.append(sampler)
            self.compiled_sampler = compiled_samplers

        # get detector coordinates:
        detector_coordinates = circuit.get_detector_coordinates()
        # get coordinates of detectors (divide by 2 because stim labels 2d grid points)
        # coordinates are of type (d_west, d_north, hence the reversed order)
        detector_coordinates = np.array(list(detector_coordinates.values()))
        # rescale space like coordinates:
        detector_coordinates[:, : 2] = detector_coordinates[:, : 2] / 2
        # convert to integers
        self.detector_coordinates = detector_coordinates.astype(np.uint8)

        # syndrome mask
        sz = self.code_size + 1
        syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
        syndrome_x[::2, 1 : sz - 1 : 2] = 1
        syndrome_x[1::2, 2::2] = 1
        syndrome_z = np.rot90(syndrome_x) * 3
        self.syndrome_mask = np.dstack([syndrome_x + syndrome_z] * (self.d_t + 1))
    
    def stim_to_syndrome_3D(self, detection_events_list):
        '''
        Converts a stim detection event array to a syndrome grid. 
        1 indicates a violated X-stabilizer, 3 a violated Z stabilizer.
        '''
        mask = np.repeat(self.syndrome_mask[None, ...], detection_events_list.shape[0], 0)
        syndrome_3D = np.zeros_like(mask)
        syndrome_3D[:, self.detector_coordinates[:, 1], self.detector_coordinates[:, 0],
                    self.detector_coordinates[:, 2]] = detection_events_list

        # convert X (Z) stabilizers to 1(3) entries in the matrix
        syndrome_3D[np.nonzero(syndrome_3D)] = mask[np.nonzero(syndrome_3D)]
        # return as [n_shots, x_coordinate, z_coordinate, time]
        return syndrome_3D

    def get_batch_of_graphs(self, syndromes):
        syndromes = self.stim_to_syndrome_3D(syndromes)
        syndromes = syndromes.astype(np.float32)
        defect_inds = np.nonzero(syndromes)
        defects = syndromes[defect_inds]

        defect_inds = np.transpose(np.array(defect_inds))

        x_defects = defects == 1
        z_defects = defects == 3

        node_features = np.zeros((defects.shape[0], 6), dtype=np.float32)

        node_features[x_defects, 0] = 1
        node_features[x_defects, 2:] = defect_inds[x_defects, ...]
        node_features[z_defects, 1] = 1
        node_features[z_defects, 2:] = defect_inds[z_defects, ...]

        node_features.max(axis=0)
        x_cols = [0, 1, 3, 4, 5]
        batch_col = 2

        x = torch.tensor(node_features[:, x_cols]).to(self.device)
        batch_labels = torch.tensor(node_features[:, batch_col]).long().to(self.device)
        pos = x[:, 2:]

            # get edge indices
        edge_index = knn_graph(pos, self.m_nearest_nodes, batch=batch_labels)

        # compute distances to get edge attributes
        dist = torch.sqrt(((pos[edge_index[0], :] - pos[edge_index[1], :])**2).sum(dim=1, keepdim=True))
        edge_attr = 1 / dist ** self.power

        return x, edge_index, batch_labels, edge_attr

    def evaluate_test_set(self, x, edge_index, batch_label, edge_attr, flips, 
                          n_trivial_syndromes, loss_fun, n_samples):
        correct_preds = 0

        # loop over batches
        with torch.no_grad():
            out = self.model(x, edge_index, batch_label, edge_attr)

            prediction = (self.sigmoid(out.detach()) > 0.5).long()
            target = flips.long()
            correct_preds += int(((prediction == target).sum(dim=1) == 
                                  self.model_settings["num_classes"]).sum().item())
        val_loss = loss_fun(out, flips)
        val_accuracy = (correct_preds + n_trivial_syndromes) / n_samples
        return val_loss, val_accuracy

    def train(self):
        time_start = time.perf_counter()
        # training settings
        current_epoch = self.epoch
        n_epochs = self.training_settings["epochs"]
        dataset_size = self.training_settings["dataset_size"]
        batch_size = self.training_settings["batch_size"]
        validation_set_size = self.training_settings["validation_set_size"]
        test_set_size = self.training_settings["test_set_size"]
        n_batches = dataset_size // batch_size
        loss_fun = torch.nn.BCEWithLogitsLoss()
        sigmoid = torch.nn.Sigmoid()

        # Learning rate:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.training_settings["lr"]

        # generate test set
        self.initialise_simulations(self.test_error_rate)
        test_syndromes, test_observable_flips, test_n_trivial_syndromes = \
            sample_syndromes(test_set_size, self.compiled_sampler, self.device)
        test_x, test_edge_index, test_batch_labels, test_edge_attr = self.get_batch_of_graphs(test_syndromes)

        # TIMING:
        time_sample = 0.
        time_fit = 0.
        time_write = 0.

        # generate train set:
        print(f'Running with {n_batches} batches of size {batch_size} per epoch.')
        self.initialise_simulations(self.train_error_rate)

        # generate validation set
        self.initialise_simulations(self.train_error_rate)
        # the complete validation set containing batch_size data points:
        syndromes, val_flips, n_trivial = sample_syndromes(validation_set_size, self.compiled_sampler, self.device)
        val_x, val_edge_index, val_batch_labels, val_edge_attr = self.get_batch_of_graphs(syndromes)

        # INITIALIZE WANDBE
        if self.wandb_log:
            wandb.init(project="surface_codes", name = self.save_name, config = {
                **self.model_settings, **self.graph_settings, **self.training_settings})
    
        for epoch in range(current_epoch, n_epochs):
            train_loss = 0
            epoch_n_graphs = 0
            epoch_n_correct = 0
            for j in range(n_batches):
                # the complete dataset containing batch_size data points:
                sample_start = time.perf_counter()
                syndromes, flips, n_trivial = sample_syndromes(batch_size, self.compiled_sampler, self.device)
                x, edge_index, batch_labels, edge_attr = self.get_batch_of_graphs(syndromes)
                sample_end = time.perf_counter()
                time_sample += (sample_end - sample_start)
                # forward/backward pass
                fit_start = time.perf_counter()
                self.optimizer.zero_grad()
                out = self.model(x, edge_index, batch_labels, edge_attr)
                loss = loss_fun(out, flips)
                loss.backward()
                self.optimizer.step()

                # update loss and accuracies
                prediction = (sigmoid(out.detach()) > 0.5).long()
                target = flips.long()
                epoch_n_correct += int(((prediction == target).sum(dim=1) == 
                            self.model_settings["num_classes"]).sum().item())
                train_loss += loss.item() * batch_size
                epoch_n_graphs += batch_size
                fit_end = time.perf_counter()
                time_fit += (fit_end - fit_start)
            
            # train accuracy 
            train_loss /= epoch_n_graphs
            train_accuracy = epoch_n_correct / (batch_size * n_batches)

            # validation (set the n_trivial syndromes to 0)
            val_loss, val_accuracy = self.evaluate_test_set(val_x, val_edge_index,
                                                        val_batch_labels, val_edge_attr,
                                                        val_flips,
                                                        0,
                                                        loss_fun,
                                                        validation_set_size)
            
            # test
            test_loss, test_accuracy = self.evaluate_test_set(test_x, test_edge_index,
                                                            test_batch_labels, test_edge_attr,
                                                            test_observable_flips,
                                                            test_n_trivial_syndromes,
                                                            loss_fun,
                                                            test_set_size)

            print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')


            # save training attributes after every epoch
            self.training_history["epoch"] = epoch
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["test_loss"].append(test_loss)
            self.training_history["train_accuracy"].append(train_accuracy)
            self.training_history["val_accuracy"].append(val_accuracy)
            self.training_history["test_accuracy"].append(test_accuracy)
            self.save_model_w_training_settings()
            
            write_start = time.perf_counter()
            # Log training and testing metrics to wandb
            if self.wandb_log:
                metrics = {'loss': train_loss, 'accuracy': train_accuracy, 
                           'validation accuracy': val_accuracy,
                           'test accuracy': test_accuracy}
                wandb.log(metrics)
            write_end = time.perf_counter()
            time_write += (write_end - write_start)
        
        runtime = time.perf_counter()-time_start
        print('Training completed after {:.1f}:{:.1f}:{:.1f}'.format(*divmod(divmod(
            runtime, 60)[0], 60), *divmod(runtime, 60)[::-1]))
        
        print(f'Sampling and Graphing: {time_sample:.0f}s')
        print(f'Fitting: {time_fit:.0f}s')
        print(f'Writing: {time_write:.0f}s')
    
    def test(self):
        print("==============TESTING==============")
        time_start = time.perf_counter()
        loss_fun = torch.nn.BCEWithLogitsLoss()

        self.initialise_simulations(self.test_error_rate)
        batch_size = self.training_settings["acc_test_batch_size"]
        n_test_batches = self.training_settings["acc_test_size"] // batch_size

        test_accuracy = 0
        n_trivial_syndromes = 0
        for i in range(n_test_batches):
            # generate test set
            syndromes, observable_flips, n_trivial_syndromes = \
                sample_syndromes(batch_size, self.compiled_sampler, self.device)
            val_x, val_edge_index, val_batch_labels, val_edge_attr = self.get_batch_of_graphs(syndromes)
            test_loss_batch, test_accuracy_batch = self.evaluate_test_set(val_x, val_edge_index,
                                                                val_batch_labels, val_edge_attr,
                                                                observable_flips,
                                                                n_trivial_syndromes,
                                                                loss_fun,
                                                                batch_size)
            print(f'Accuracy: {test_accuracy_batch:.6f} Trivials: {n_trivial_syndromes} No. Samples: {batch_size}')
            test_accuracy += test_accuracy_batch
            n_trivial_syndromes += n_trivial_syndromes

        test_accuracy = test_accuracy / n_test_batches
        print(f'Test Acc: {test_accuracy}, tested on {n_test_batches * batch_size} '
              f'samples, of which {n_trivial_syndromes} trivial samples.')

        runtime = time.perf_counter()-time_start
        print('Testing completed after {:.1f}:{:.1f}:{:.1f}'.format(*divmod(divmod(
            runtime, 60)[0], 60), *divmod(runtime, 60)[::-1]))


    def run(self):
        if self.training_settings["resume_training"]:
            print(f'Loading model {self.saved_model_path}')
        print(f'Running on code size {self.code_size} with {self.d_t} repetitions.')
        if self.training_settings['run_training']:
            if self.training_settings['run_test']:
                self.train()
                self.test()
                # only save final test accuracy if trained before
                self.save_model_w_training_settings()
            else:
                self.train()
        else:
            self.test()