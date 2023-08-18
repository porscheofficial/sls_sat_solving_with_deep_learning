# GNN-based oracle factories for SLS solvers.

This is the code to accmpany the submission "Using deep learning to construct Stochastic Local Search SAT solvers with performance bounds". 

## Setup
0. Clone the repository

Clone the repository via
'''ruby
git clone <repo>
'''
and navigate to the repository using
'''ruby
cd <repo>
'''

1. Create virtual environment and install all the requirements in the virtual environment

To start with, install virtualenv (if you have not done so already) by running
'''ruby
pip install virtualenv
'''
Run 
'''ruby
virtualenv venv
'''
to create a new virtual environment environment and activate it by typing
'''ruby
source venv/bin/activate
'''
Now we can install the requirements in the current environment. To do so, type
'''ruby
pip install -r requirements.txt
'''

Note that if you want to use a GPU for training, you should run
'''
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
'''
to install the GPU version of jax. In case of a TPU, use the following command
'''
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
'''


2. Compile the rust code

The MT algorithm and WalkSAT are implemented in Rust because of performance reasons. Start by installing Rust by typing
'''ruby
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
'''
and run
'''ruby
maturin develop -r
'''

3. (optional) run tests

## Usage

To replicate the experiments, you can either take the trained models or rerun the training. 

To use the existing models, simply run the tutorial notebook. 

To train, run the "run all experiments" config file, and then point the tutorial to those model files saved in there. You start running the experiments by typing 
'''
 chmod +x run_all_experiments.sh 
'''
and then
'''
./run_all_experiments.sh 
'''