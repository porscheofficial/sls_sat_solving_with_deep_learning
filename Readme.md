# GNN-based oracle factories for SLS solvers.

This is the code to accompany the submission "Using deep learning to construct Stochastic Local Search SAT solvers with performance bounds". 

## Setup
<!-- **0. Clone the repository**

Clone the repository via

`
git clone <repo>
`

and navigate to the repository using

`
cd <repo>
` -->

**1. Create virtual environment and install all the requirements in the virtual environment**

This app has been run and tested with Python 3.10. To start with, ensure you have a version of Python3.10 installed locally. To install virtualenv

`
pip3.10 install virtualenv
`

Run

`
virtualenv venv
`
or 
`
python3.10 -m virtualenv venv
`

to create a new virtual environment environment and activate it by typing

`
source venv/bin/activate
`

Now we can install the requirements in the current environment. To do so, type

`
pip install -r requirements.txt
`

Note that if you want to use a GPU for training, you should run

`
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
`

afterwards to install the GPU version of Jax. If you use a TPU, use the following command:

`
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
`


**2. Compile the rust code**

The MT algorithm and WalkSAT are implemented in Rust (we used rustc version 1.66.0) because of performance reasons. Start by installing Rust via rustup by typing

`
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
`

and run

`
maturin develop -r
`

If you experience problems with the compilation process, check the compiler version (using `rustc -V`) and try downgrading (`rustup install 1.66`).

**3. (optional) run tests**

There are dedicated test files for the code. To run them, type

`
pytest
`

## Usage

### Files from document

The plots from the experiments can be found under `Data/plots`, the pre-trained models


### Run GNN-boosted SLS solvers


### Using existing model

To replicate the experiments, you can either **take the trained models or rerun the training**. 

1. To **use the existing models**, simply run the tutorial notebook. 

### Using retrained models

1. Run the "run_all_experiments" config file by typing 

`
 chmod +x run_all_experiments.sh 
`

and then

`
./run_all_experiments.sh 
`

2. Point the tutorial notebook to those model files saved in there (by default, the model files will be stored under "experiments/params_save"). You can do this in the first cell of the tutorial notebook. 

### Generating datasets

We provide the dataset used for the experiments reported in the submission alongside the code. However, to genereate new datasets, use the notebook  `python/src/instace_sampling.ipynb`.