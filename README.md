# Graph Decoder
Graph neural network decoder for the rotated surface code. 

This repository accompanies the article:

[Data-driven decoding of quantum error correcting codes using graph neural networks](https://arxiv.org/abs/2307.01241)

If you use this code in your research, please cite:

@article{lange2025data,

  title={Data-driven decoding of quantum error correcting codes using graph neural networks},
  
  author={Lange, Moritz and Havstr{\"o}m, Pontus and Srivastava, Basudha and Bengtsson, Isak and Bergentall, Valdemar and Hammar, Karl and Heuts, Olivia and van Nieuwenburg, Evert and Granath, Mats},
  
  journal={Physical Review Research},
  
  volume={7},
  
  number={2},
  
  pages={023181},
  
  year={2025},
  
  publisher={APS}
}

Includes the source code for the GNN decoder, as well as scripts to run the code on a cluster.

## Getting started
Follow these steps to run the code on a cluster.

* Clone the repository with
`git clone https://github.com/LangeMoritz/GNN_decoder`
* Install the required packages in `requirements.txt`, for example using a virtual environment:
```
python3 -m venv .venv 
source .venv/bin/activate
python -m pip install -r requirements.txt
```
* Install PyTorch by following the instructions found [here](https://pytorch.org/get-started/locally/).
* Install PyTorch Geometric by following the instructions found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Repository structure
* `src/` contains the "source code" for the project, split into a number of modules:
  * `decoder.py` A class for creating a decoder object with methods for training a GNN decoder and running decoding simulations.
  * `gnn_models.py` Contains PyTorch geometric graph neural network decoder models.
  * `graph_representation.py` Functions for creating syndromes graphs.
  * `utils.py` Function for reading argument files.
```
├── src
│   ├── decoder.py
│   ├── gnn_models.py
│   ├── graph_representation.py
│   └── utils.py
```
* `models/` contains trained models corresponding to fig. 3 (`circuit_level_noise/`), fig. 5 and 6 (`repetition_code/`) and fig. 7 (`perfect_stabilizers/`)
* `run_training.sh` is the shell run script used to start is jobs for `train_nn.py`.
* `.gitignore` lists files and directories in the git repository to be ignored in commits.
* `requirements.txt` lists the required python packages. See the Getting Started section above.
  
