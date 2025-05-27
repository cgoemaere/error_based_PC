
# Error Optimization for Predictive Coding

_Official codebase for "**Error Optimization: Overcoming Exponential Signal Decay in Deep Predictive Coding Networks**"_ [[arXiv]](https://arxiv.org/abs/2505.20137)

[![arXiv](https://img.shields.io/badge/arXiv-2505.20137-b31b1b.svg)](https://arxiv.org/abs/2505.20137)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-purple.svg?logo=PyTorch%20Lightning)](https://github.com/Lightning-AI/pytorch-lightning)
[![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://github.com/wandb/wandb)

---

## 👉 What Error Optimization can do for you

On digital hardware, traditional Predictive Coding hits a wall with deep networks due to exponential signal decay.  
Error Optimization fixes that:

| Traditional state-based PC | Error Optimization |
|---|---|
| 🚫 Struggles with depth (4-5 layers max) | ✅ **Scales to any depth you want** |
| 🐌 Slow convergence (thousands of iterations) | ⚡ **100-1000x faster convergence** |
| 📉 Poor performance on deep networks | 🎯 **Matches backprop performance** |
| 📃 Energy optimization, local weight updates | 📋 **Preserves PC's key properties** |
| 🧠 Meant for neuromorphic hardware | 🖥️ **Designed for GPUs** |

**Bottom line**: If you're working with Predictive Coding on digital hardware, Error Optimization is what you need.

## ⚡ Quickstart

1.  **Install dependencies via Conda**
	```bash
	# Clone and setup (2 minutes)
	git clone https://github.com/cgoemaere/pc_error_optimization
	cd pc_error_optimization/
	conda create --name error_optim_test_env --file requirements.txt -c conda-forge -c pytorch
	conda activate error_optim_test_env

	# Sanity check
	python3 -c "from pc_e import PCE; print('Installation successful!')"
	```

2.  **Play around with the interactive notebook**  
Launch `PredictiveCodingPlayground.ipynb`  in Jupyter for hands-on experimentation with PC and EO.

## 📂 Repository structure

```code
> main branch              # Code for MNIST & FashionMNIST
├── sweep.py               # Launches experiment sweeps (reproduce paper results)
├── requirements.txt       # All dependencies (see below)
├── PredictiveCodingPlayground.ipynb  # Notebook for interactive exploration
├── datamodules/           # PyTorch Lightning datamodules (datasets, augmentations)
├── get_arch.py            # Model architectures
├── pc_e.py                # Core PC formulation
└── get_variants.py        # All loss/optimization variants (SO, EO, BP)

> cifar branch             # Same as above, for CIFAR-10/100 (with VGG/ResNet)
├── configs_results/       # YAML files with final training configurations
├── configs_sweeps/        # YAML files with hyperparameter sweep settings
├── {arch}_{algo}.py       # Dedicated file per architecture / algorithm (instead of sweep.py)
└── ...
```

## 📊 Reproducing our results

- **Reproduce MNIST/FashionMNIST experiments**
	```bash
	# Run a sweep (prepend with 'nohup' to run in background)
	python3 sweep.py
	```

	The sweep works with a config file to select your desired setup:
	```python
	sweep_configuration = {
       "method": "grid",
       "metric": {"goal": "maximize", "name": "val_acc"},
       "parameters": {
           "dataset": {"value": "FashionMNIST"},  # or "MNIST"
           "algorithm": {"value": "EO"},          # "SO", "EO", or "BP"
           "USE_DEEP_MLP": {"values": [True]},    # 20 layers vs 4
           "e_lr": {"values": [0.1, 0.3]},        # state/error learning rate
           "iters": {"values": [64, 256]},        # nr of optimization steps
           ...
       },
	}
	```
	_(see Appendix D for all hyperparameter settings used in our experiments)_

- 	 **Reproduce MNIST figures**
	See `mnist_poc` folder:
		- Fig. 1: `fig1.ipynb`
		- Fig. B.1: `fig1_float64_binomial.ipynb`
		- Fig. 5: `analysis_deep_linear.ipynb`
		- Fig. C.1: `analysis_deep_MLP.ipynb`
	

-  **Reproduce CIFAR-10/100 experiments**
	```bash
	git checkout cifar
	python3 vgg_pce.py  # or resnet_bp.py or ...
	```

_All experiment configs, logging, and results generation go directly to wandb._

## 📚 Citation

If our work helps your research, please cite:
```bibTeX
@article{goemaere2025error_optim,
	title = {Error Optimization: Overcoming Exponential Signal Decay in Deep Predictive Coding Networks},
	author = {C\'edric Goemaere and Gaspard Oliviers and Rafal Bogacz and Thomas Demeester},
	year = {2025},
	journal = {arXiv preprint arXiv: 2505.20137}
}
```
