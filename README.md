# Code for "When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds"

This repository contains the official implementation for the paper *"When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds"*.

The primary objective of this repository is to facilitate the exact reproduction of the experimental results presented in the paper. It is highly optimized for reproducibility rather than active development.

## Repository Structure

The repository is organized into three main directories:

1. `resnet_expt/`: Contains the implementation for training ResNet-20 on the CIFAR-10 dataset.
2. `nanoGPT_expts/`: Contains the scripts for training nanoGPT on the C4 dataset and estimating noise density ($\phi$).
3. `toy_problem/`: Contains the simplified constructed examples demonstrating the robustness of SignSGD over standard SGD, used to generate Figure X [Please replace X with actual figure number].

------

## 1. Environment Setup

To begin reproducing our results, please follow these steps to set up the environment:

**Step 1:** Clone the repository and navigate to the root directory.

**Step 2:** Create a clean Conda environment and install the required dependencies:

Bash

```
conda create -n signsgd_env python=3.10 -y
conda activate signsgd_env
pip install -r requirements.txt
```

------

## 2. Toy Models

The files `deter.py` and `sto.py` correspond to the two meticulously crafted toy examples (deterministic and stochastic, respectively) discussed in the paper.

To execute the scripts and generate the corresponding results, run:

Bash

```
cd toy_problem
python deter.py
python sto.py
```

------

## 3. ResNet Experiments

Our CNN implementations and hyper-parameter configurations closely follow the standard setup established in the original [SignSGD paper](https://github.com/jxbz/signSGD). Notably, we have entirely refactored the original `MXNet` implementation into a modern `PyTorch` version, ensuring significantly faster execution on contemporary GPUs.

**To reproduce the CNN results:**

**Step 1:** Run the training script to collect the gradient density data. This script supports distributed training:

Bash

```
cd resnet_expt
torchrun --nproc_per_node=4 run_expts.py --optim sgd
```

**Step 2:** Once the training finishes, locate the generated history CSV file. Update the `CSV_FILE` variable in `show_cnn_sparse.py` with your file path, and execute the plotting script:

Bash

```
python show_cnn_sparse.py
```

------

## 4. nanoGPT Experiments

**Step 1: Configuration**

Before running the scripts, please update the global variables `CONDA`, `results_base_folder`, and `datasets_dir` in the bash scripts with your actual local paths.

**Step 2: Training and Comparison**

To train both standard SGD and SignSGD on the C4 dataset, execute:

Bash

```
cd nanoGPT_expts
bash signsgd.sh
```

Upon completion, update the dictionary in `plot/show_compare.py` with the paths to your newly generated result directories:

Python

```
experiments = {
    "SGD": "path/to/your/SGD/result/folder",
    "SignSGD": "path/to/your/SignSGD/result/folder",
}
```

Then, generate the comparative loss curves (PDF):

Bash

```
cd plot
python show_compare.py
```

**Step 3: Noise Sparsity Visualization**

To analyze the gradient noise sparsity evolution during LLM training, run the noise tracking script:

Bash

```
bash noise.sh
```

Update the CSV file path in `plot/show_llm_sparse.py` to point to your tracked density file, then generate the visualization:

Bash

```
python show_llm_sparse.py
```

> **Note on Sparsity Sampling:** > By default, the `noise.sh` script employs a three-phase sampling strategy (tracking noise only at the start, middle, and end of training for a few steps each) to minimize computational overhead. This is sufficient to demonstrate our core finding that noise becomes increasingly sparse over time.
>
> If you wish to reproduce the continuous, full-stage sparsity evolution exactly as presented in our paper's figure, please change the `sample_itr` parameter in the script to match the total `ITERATIONS`. The script will then record the density metrics continuously across the entire training procedure.

## Reference
If you find our work helpful, feel free to cite

```
@article{tao2026when,
  title={{When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds}},
  author={Tao, Hongyi and Yu, Dingzhi and Zhang, Lijun},
  journal={arXiv preprint arXiv:2605.06615},
  year={2026}
}
```
