# DeepSSE: Deep Learing based DOA estimator for variable and unkown number of signals

This repository includes the source code used in our recent paper:

Qian Xu, Yulong Gao, Ruoyu Zhang, Jinshan Kong and Chau Yuen. "Deep Learning-Based DOA Estimation via Grid Search Approach for Unknown and Variable Number of Signals"

## Abstract

Deep learning-based direction-of-arrival (DOA) estimators have displayed superior performance to classical model-based DOA estimators in many scenarios. However, most deep learning-based DOA estimators can only be used for a fixed signal number or pre-trained signal number ranges. This makes deep learning-based DOA estimators not practical and flexible for real-world applications. In this paper, we propose a deep learning-based spatial spectrum estimator (DeepSSE). It demonstrates remarkable generalization capability with unknown and variable number of signals, not limited by the maximum signal number present in the training data. This capability is achieved by leveraging a novel angular grid search (AGS) process and the asymmetric loss (ASL). The AGS, which emulates that of the multiple signal classification (MUSIC) algorithm, enables DeepSSE to detect as many target DOAs as possible. And the ASL enhances the performance of DeepSSE under multiple signals by overcoming the imbalance of positive and negative angular grids. Furthermore, we introduce the optimal sub-pattern assignment (OSPA) metric to DOA estimation for the first time, to address the lack of performance evaluation metrics in scenarios with a variable number of signals. Extensive numerical results demonstrate that DeepSSE outperforms other DOA estimators across various scenarios, especially when the signal numbers are far beyond the maximum number in training set.

## Getting Started

First clone this repository.

```python
git clone https://github.com/zhiim/deepsse.git
```

Init the submodule.

> [!NOTE]  
> [doa_py](https://github.com/zhiim/doa_py) is embedded as git submodule. But do not pull new commit, as there may be incompatibility.

```python
git submodule init
git submodule update
```

[uv](https://github.com/astral-sh/uv) is recommended to manage this project.

```python
uv venv
uv sync
```

## How to Training

### Obtain Datasets

- Download the original dataset used in our paper from [here](https://drive.google.com/drive/folders/1cK2AikE1b8V72EVCZ-yZk_MaitZgKVRl?usp=drive_link).
- Generate dataset yourself

```python
# for 1, 2, 3, 4 signal numbers
python gen_data.py data/1.yaml
python gen_data.py data/2.yaml
python gen_data.py data/3.yaml
python gen_data.py data/4.yaml
```

### Train the model

```python
python train.py -c config.yaml
```

## License

This project is licensed under the [MIT](LICENSE) License - see the LICENSE file for details.

## Acknowledgements

Some code references [PyTorch Template Project](https://github.com/victoresque/pytorch-template) and [DA-MUSIC](https://github.com/DA-MUSIC/TVT23).
