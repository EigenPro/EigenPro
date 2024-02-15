# EigenPro
EigenPro [[1-3]](#References) is a GPU-enabled fast and scalable solver for training kernel machines.
It applies a projected stochastic gradient method with dual preconditioning to enable major speed-ups.
It is currently based on a PyTorch backend.

## Highlights
- *Fast*: EigenPro is the fastest kernel method at large scale.
- *Plug-and-play*: Our method learns a quality model with little hyper-parameter tuning in most cases.
- *Scalable*: The training time of one epoch is nearly linear in both model size and data size. This is the first kernel method that achieves such scalability without any compromise on testing performance.

## Coming Soon
- *Support for multi-GPU and model-parallelism*: We are adding support for multiple GPUs and model-parallelism.
---

## Usage

### Installation
```
pip install git+ssh://git@github.com/EigenPro/EigenPro.git@main
```

### Run Example
Linux:
```
bash examples/run_fmnist.sh
```
Windows:
```
examples\run_fmnist.bat
```

Jupyter Notebook:
[examples/notebook.ipynb](https://github.com/EigenPro/EigenPro/blob/main/examples/notebook.ipynb)


See files under `examples/` for more details.


## Empirical Results
In the experiments described below, `P` denotes the number of centers (model size), essentially representing the model size, while 'd' signifies the ambient dimension. For all experiments, a Laplacian kernel with a bandwidth of 20.0 was employed.

### 1. CIFAR5M Extracted Features on single GPU

We used extracted features from the pretrained 'mobilenet-2' network available in the timm library. The benchmarks processed the full **5 million samples** of CIFAR5M with **d = 1280** for **one epoch** for two versions of EigenPro and FALKON [[4-6]](#References).
All of these experiments were run on a **single A100** GPU. The maximum RAM we had access to was 1.2TB, which was not sufficient for FALKON with 1M centers.

| Method      | P = 64k        |        | P = 128k       |        | P = 256k       |        | P = 512k       |        | P = 1024k      |         |
|-------------|----------------|--------|----------------|--------|----------------|--------|----------------|--------|----------------|---------|
|             | Accuracy          | Time   | Accuracy          | Time   | Accuracy          | Time   | Accuracy          | Time   | Accuracy          | Time    |
| **EigenPro (latest)** | 87.99%         | 271s   | 88.25%         | 309s   | 88.43%         | 406s   | 88.58%         | 695s   | 88.74%         | 1268s   |
| EigenPro 3.0 [[1]](#References) | 88.33%         | 1359s  | 88.42%         | 3014s  | 88.61%         | 7663s  | 88.56%         | 21845s | > 24hrs        | -       |
| FALKON [[4-6]](#References) | 86.09%         | 184s   | 86.55%         | 537s   | 86.73%         | 2308s  | 86.71%         | 14433s | out of memory  | -       |



### 2. Libri speach Extracted Features on single GPU

We used **10 million samples** with **d = 1024** for **one epoch** for two versions of EigenPro and FALKON. All of these experiments were run on a **single V100** GPU. The maximum RAM available for this experiment was 300GB, which was not sufficient for FALKON with more than 128K centers. The features are extracted using an acoustic model (a VGG+BLSTM architecture in [[7]](#References)) to align the length of audio and text.

| Method      | P = 64k         |            | P = 128k        |            | P = 256k        |            | P = 512k        |            | P = 1024k       |            |
|-------------|-----------------|------------|-----------------|------------|-----------------|------------|-----------------|------------|-----------------|------------|
|             | Accuracy          | Time       | Accuracy           | Time       |Accuracy           | Time       | Accuracy           | Time       | Accuracy          | Time       |
| **EigenPro (latest)**  | 86.84%          | 980s       | 87.80%          | 1157s      | 88.33%          | 1440s      | 88.89%          | 2185s      | 89.49%          | 4229s      |
| EigenPro 3.0 [[1]](#References) | 85.43%          | 8697s      | 84.75%          | 20492s     | > 18hrs         | -          | > 24hrs         | -          | > 24hrs         | -          |
| FALKON [[4-6]](#References)      | 81.04%          | 535s       | 82.30%          | 1290s      | out of mem   | -          | out of mem   | -          | out of mem   | -          |

---

# References
1. Abedsoltan, Amirhesam and Belkin, Mikhail and Pandit, Parthe, "Toward Large Kernel Models," Proceedings of the 40th International Conference on Machine Learning, ICML'23, JMLR.org, 2023. [Link](https://proceedings.mlr.press/v202/abedsoltan23a/abedsoltan23a.pdf)
2. Siyuan Ma, Mikhail Belkin, "Kernel machines that adapt to GPUs for effective large batch training," Proceedings of the 2nd SysMLConference, 2019. [Link](https://mlsys.org/Conferences/2019/doc/2019/171.pdf)
3. Siyuan Ma, Mikhail Belkin, "Diving into the shallows: a computational perspective on large-scale shallow learning," Advances in Neural Information Processing Systems 30 (NeurIPS 2017). [Link](https://proceedings.neurips.cc/paper_files/paper/2017/file/bf424cb7b0dea050a42b9739eb261a3a-Paper.pdf)
4. Giacomo Meanti, Luigi Carratino, Lorenzo Rosasco, Alessandro Rudi, “Kernel methods through the roof: handling billions of points efficiently,” Advances in Neural Information Processing Systems, 2020. [Link](https://proceedings.neurips.cc/paper_files/paper/2020/file/a59afb1b7d82ec353921a55c579ee26d-Paper.pdf)
5. Alessandro Rudi, Luigi Carratino, Lorenzo Rosasco, “FALKON: An optimal large scale kernel method,” Advances in Neural Information Processing Systems, 2017. [Link](https://papers.nips.cc/paper_files/paper/2017/file/05546b0e38ab9175cd905eebcc6ebb76-Paper.pdf)
6. Ulysse Marteau-Ferey, Francis Bach, Alessandro Rudi, “Globally Convergent Newton Methods for Ill-conditioned Generalized Self-concordant Losses,” Advances in Neural Information Processing Systems, 2019. [Link](https://arxiv.org/pdf/1907.01771.pdf)
7. Hui, L. and Belkin, M. "Evaluation of Neural Architectures Trained with Square Loss vs Cross-Entropy in Classification Tasks." In International Conference on Learning Representations, 2021. [Link](https://arxiv.org/abs/2006.07322)

# Cite us
