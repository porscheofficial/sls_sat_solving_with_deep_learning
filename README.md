# SLS based SAT solving with deep learning and performance bounds.

This is the code to accompany the publication "Using deep learning to construct Stochastic Local Search SAT solvers with performance bounds" available at https://arxiv.org/abs/2309.11452

## Abstract

The Boolean Satisfiability problem (SAT) is the most prototypical NP-complete problem and of great practical relevance. One important class of solvers for this problem are stochastic local search (SLS) algorithms that iteratively and randomly update a candidate assignment. Recent breakthrough results in theoretical computer science have established sufficient conditions under which SLS solvers are guaranteed to efficiently solve a SAT instance, provided they have access to suitable "oracles" that provide samples from an instance-specific distribution, exploiting an instance's local structure. Motivated by these results and the well established ability of neural networks to learn common structure in large datasets, in this work, we train oracles using Graph Neural Networks and evaluate them on two SLS solvers on random SAT instances of varying difficulty. We find that access to GNN-based oracles significantly boosts the performance of both solvers, allowing them, on average, to solve 17% more difficult instances (as measured by the ratio between clauses and variables), and to do so in 35% fewer steps, with  improvements in the median number of steps of up to a factor of 8. As such, this work bridges formal results from theoretical computer science and practically motivated research on deep learning for constraint satisfaction problems and establishes the promise of purpose-trained SAT solvers with performance guarantees.

## General idea

<img width="1000" alt="oracle_idea" src="https://github.com/porscheofficial/sls_sat_solving_with_deep_learning/assets/105794634/fb115e3b-c829-49cd-9d97-96eaeaf18948">

The figure above illustrates the general idea of this work. Left: A simple SLS solver finds a solution to a SAT instance by repeatedly and randomly updating a small subset of the variables. Middle: An oracle-based SLS solver uses samples from an oracle $O$ that is provided as part of the input to update the variables at each iteration. Right: We use a deep learning model to train an oracle factory $F_θ$ that maps an incoming instance to an oracle which is then fed into an oracle-based. This approach is motivated by results that provide sufficient conditions for an oracle-based SLS solver to find a solution efficiently, based on properties of the oracle.

The trained oracle can be used in various SLS solvers. In case you are interested, feel free to play around and see how the introduction of an oracle leads to better performing solvers! 🚀🚀🚀

## Setup
**0. Clone the repository**

Clone the repository via

`
git clone https://github.com/porscheofficial/sls_sat_solving_with_deep_learning.git
`

and navigate to the repository using

`
cd sls_sat_solving_with_deep_learning
`

**1. Create virtual environment and install all the requirements in the virtual environment**

This app has been run and tested with Python 3.10. To start with, ensure you have a version of Python3.10 installed locally. To install virtualenv type

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

1. To **use the existing models**, simply run the tutorial notebook at `python/src/tutorial.ipynb`. 

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

## How to cite

Please consider citing our paper if you use our code in your project.

Maximilian J. Kramer, Paul Boes. (2023). ["Using deep learning to construct Stochastic Local Search SAT solvers with performance bounds"](https://arxiv.org/abs/2309.11452). arXiv preprint arXiv:2309.11452

```
@misc{kramer2023,
      title={Using deep learning to construct stochastic local search SAT solvers with performance bounds}, 
      author={Maximilian Kramer and Paul Boes},
      year={2023},
      eprint={2309.11452},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Contributing

This GNN-based oracle factory is openly developed in the wild and contributions (both internal and external) are highly appreciated.
See [CONTRIBUTING.md](./CONTRIBUTING.md) on how to get started.

If you have feedback or want to propose a new feature, please [open an issue](https://github.com/porscheofficial/sls_sat_solving_with_deep_learning/issues).
Thank you! 😊

## Acknowledgements

This project is part of the AI research of [Porsche Digital](https://www.porsche.digital/). ✨


## License

Copyright © 2023 Porsche Digital GmbH

Porsche Digital GmbH publishes this open source software and accompanied documentation (if any) subject to the terms of the [MIT license](./LICENSE.md). All rights not explicitly granted to you under the MIT license remain the sole and exclusive property of Porsche Digital GmbH.

Apart from the software and documentation described above, the texts, images, graphics, animations, video and audio files as well as all other contents on this website are subject to the legal provisions of copyright law and, where applicable, other intellectual property rights. The aforementioned proprietary content of this website may not be duplicated, distributed, reproduced, made publicly accessible or otherwise used without the prior consent of the right holder.
