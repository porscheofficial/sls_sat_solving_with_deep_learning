# SLS based SAT solving with deep learning and performance bounds.

This is the code to accompany the project "Using deep learning to construct Stochastic Local Search SAT solvers with performance bounds".

## Abstract

The Boolean Satisfiability problem (SAT), as the prototypical NP-complete problem, is crucial both in theoretical computer science and for practical applications. 
For this problem, stochastic local search (SLS) algorithms, which iteratively and randomly update candidate assignments, present an important and theoretically well-studied class of solvers. Recent theoretical advancements have identified conditions under which SLS solvers efficiently solve SAT instances, provided they have access to suitable 'oracles', i.e., instance-specific distribution samples. We propose leveraging machine learning models, particularly graph neural networks (GNN), as oracles to enhance the performance of SLS solvers. Our approach, evaluated on random and pseudo-industrial SAT instances, demonstrates a significant performance improvement regarding step counts and solved instances. Our work bridges theoretical results and practical applications, highlighting the potential of purpose-trained SAT solvers with performance guarantees.

## General idea

![oracle_SLS_idea](oracle_SLS_idea.png)

The figure above illustrates the general idea of this work. Left: A simple SLS solver finds a solution to a SAT instance by repeatedly and randomly updating a small subset of the variables. Middle: An oracle-based SLS solver uses samples from an oracle $O$ that is provided as part of the input to update the variables at each iteration. Right: We use a deep learning model to train an oracle factory $F_θ$ that maps an incoming instance to an oracle which is then fed into an oracle-based. This approach is motivated by results that provide sufficient conditions for an oracle-based SLS solver to find a solution efficiently, based on properties of the oracle.

The trained oracle can be used in various SLS solvers and on various datasets. In case you are interested, feel free to play around and see how the introduction of an oracle leads to better performing solvers! 🚀🚀🚀

## Setup
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
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
`

afterwards to install the GPU version of Jax.


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

The plots from the experiments can be found under `Data/plots`, the pre-trained models can be found under `Data/models`. In our study, we investigated two kinds of datasets:

- We created a **random 3-SAT dataset** with instances of varying difficulty. Here, our measure of difficulty is the ratio $\alpha = m/n$. The datasets can be found under `Data/random_sat_data/{train,test}`.
- As **pseudo-industrial datasets**, we have used the ones provided in the benchmark in <https://github.com/zhaoyu-li/G4SATBench> [1]. This includes instances drawn from the Community Attachment (CA) model [2] and the Popularity-Similarity (PS) model [3]. For each model, three difficulties are generated (easy, medium, hard). The various datasets can be found under `Data/G4SAT` in the corresponding sub-folders `Data/G4SAT/{easy,medium,hard}/{ps,ca}/{train,test}/sat`.

The details about the datasets can be found in the submission. For each dataset we provide a training and a test dataset.

References:

[1] Zhaoyu Li, Jinpei Guo, and Xujie Si. G4SATBench: Benchmarking and advancing SAT solving with graph neural networks, 2023

[2] Jesús Giráldez-Cru and Jordi Levy. A modularity-based random sat instances generator. In Proceedings of the 24th International Conference on Artificial Intelligence, IJCAI’15, page 1952–1958. AAAI Press, 2015.

[3] Jesús Giráldez-Cru and Jordi Levy. Locality in random sat instances. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI-17, pages 638–644, 2017. doi: 10.24963/ijcai.2017/89.

### Run GNN-boosted SLS solvers


### Using existing model

To replicate the experiments, you can either **take the trained models or rerun the training**. 

1. To **use the existing models**, simply run the tutorial notebook at `python/src/tutorial.ipynb`. 

In total, we provide three pre-trained models for the 3-SAT dataset and six pre-trained models for the pseudo-industrial datasets. They found in the folders `Data/models/random_3SAT/` and `Data/models/pseudo-industrial/`, respectively. The following table summarizes the characteristics of the models.

| Path of the pre-trained model        | Description of model                                                                                                    |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `Data/models/random_3SAT/3_SAT_LLL.npy`| model trained on random 3-SAT using only the LLL-Loss     |
| `Data/models/random_3SAT/3_SAT_Gibbs.npy`|  model trained on random 3-SAT using only the Gibbs-Loss |
| `Data/models/random_3SAT/3_SAT_Gibbs_LLL.npy`|model trained on random 3-SAT using both the LLL-Loss and the Gibbs-Loss |
| `Data/models/pseudo_industrial/g4sat_easy_ca.npy`| model trained on easy CA instances using only the Gibbs-Loss     |
| `Data/models/pseudo_industrial/g4sat_medium_ca.npy`| model trained on medium CA instances using only the Gibbs-Loss     |
| `Data/models/pseudo_industrial/g4sat_hard_ca.npy`| model trained on hard CA instances using only the Gibbs-Loss     |
| `Data/models/pseudo_industrial/g4sat_easy_ps.npy`| model trained on easy PS instances using only the Gibbs-Loss     |
| `Data/models/pseudo_industrial/g4sat_medium_ps.npy`| model trained on medium PS instances using only the Gibbs-Loss     |
| `Data/models/pseudo_industrial/g4sat_hard_ps.npy`| model trained on hard PS instances using only the Gibbs-Loss     |

The corresponding hyperparameters used for the training are specified in the corresponding config files in `experiments/configs/`.


### Using retrained models


1. For the random 3-SAT experiments:

Run the "run_all_experiments_random3SAT" config file by typing 


`
 chmod +x run_all_experiments_random3SAT.sh 
`

and then

`
./run_all_experiments_random3SAT.sh 
`

By this command, the following models are trained:

| Path of the corresponding config-file        | Description of model                                                                                                    |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `experiments/configs/random_3SAT/config_LLL.yaml`| model trained on random 3-SAT using only the LLL-Loss     |
| `experiments/configs/random_3SAT/config_Gibbs.yaml`|  model trained on random 3-SAT using only the Gibbs-Loss |
| `experiments/configs/random_3SAT/config_Gibbs_LLL.yaml`|model trained on random 3-SAT using both the LLL-Loss and the Gibbs-Loss |


2. For the experiments with the pseudo-industrial datasets

Run the "run_all_experiments_g4sat" config file by typing 


`
 chmod +x run_all_experiments_g4sat.sh 
`

and then

`
./run_all_experiments_g4sat.sh 
`

By this command, the following models are trained:

| Path of the corresponding config-file       | Description of model                                                                                                    |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `experiments/configs/g4sat/easy_ca_config_Gibbs.yaml`| model trained on easy CA instances using only the Gibbs-Loss     |
| `experiments/configs/g4sat/medium_ca_config_Gibbs.yaml`| model trained on medium CA instances using only the Gibbs-Loss     |
| `experiments/configs/g4sat/hard_ca_config_Gibbs.yaml`| model trained on hard CA instances using only the Gibbs-Loss     |
| `experiments/configs/g4sat/easy_ps_config_Gibbs.yaml`| model trained on easy PS instances using only the Gibbs-Loss     |
| `experiments/configs/g4sat/medium_ps_config_Gibbs.yaml`| model trained on medium PS instances using only the Gibbs-Loss     |
| `experiments/configs/g4sat/hard_ps_config_Gibbs.yaml`| model trained on hard PS instances using only the Gibbs-Loss     |

3. Point the tutorial notebook to those model files saved in there (by default, the model files will be stored under `experiments/params_save`).


### Generating datasets

We provide the dataset used for the experiments reported in the submission alongside the code. However, to genereate new random SAT datasets, use the notebook  `python/src/instance_sampling.ipynb`. To experiment with the pseudo-industrial instances, we refer to the repository where this benchmark was released (see <https://github.com/zhaoyu-li/G4SATBench>). This repository only stores instances that were generated using the mentioned repository.

## How to cite

Please consider citing our (not yet updated) paper if you use our code in your project.

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

This GNN-based oracle factory is openly developed in the wild and contributions (both internal and external) are highly appreciated. See [CONTRIBUTING.md](./CONTRIBUTING.md) on how to get started.

If you have feedback or want to propose a new feature, please [open an issue](https://github.com/porscheofficial/sls_sat_solving_with_deep_learning/issues).
Thank you! 😊

## Acknowledgements

The origin of this project is part of the AI research of [Porsche Digital](https://www.porsche.digital/). ✨

## License

Copyright © 2023 Porsche Digital GmbH

Porsche Digital GmbH publishes this open source software and accompanied documentation (if any) subject to the terms of the [MIT license](./LICENSE.md). All rights not explicitly granted to you under the MIT license remain the sole and exclusive property of Porsche Digital GmbH.

Apart from the software and documentation described above, the texts, images, graphics, animations, video and audio files as well as all other contents on this website are subject to the legal provisions of copyright law and, where applicable, other intellectual property rights. The aforementioned proprietary content of this website may not be duplicated, distributed, reproduced, made publicly accessible or otherwise used without the prior consent of the right holder.
