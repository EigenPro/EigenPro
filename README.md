# EigenPro++

Introducing new EigenPro version.

---

# Benchmark Results

## Performance Highlights
- **Performance Boost**: EigenPro++ offers up to **100x speed** increase from the previous version (EigenPro3.0) [[1]](#References).
- **Comparison with FALKON**: The memory requirement for FALKON fundamentally scales quadratically with the number of centers, necessitating the use of **1.2 TB RAM** for conducting experiments with 512,000 centers. In contrast, EigenPro++ requires only **200 GB of RAM**. The latest version of EigenPro has successfully addressed the slow speed issue of the previous one, making it now faster than FALKON [[4-6]](#References).

## Benchmarking Details
Benchmarks were conducted on the CIFAR5M dataset using extracted features from the pretrained 'mobilenet_2' network available in the timm library. The benchmarks processed the full 5 million samples of CIFAR5M for one epoch/iteration for all versions of EigenPro and FALKON.

![Performance Comparison Table](benchmark.png)

---

# Tutorial: How to Use This Repository

Follow these steps to get started with this repository:

1. **Data Preparation**
   - Replace `X`, `Y`, `X_val`, and `Y_val` in `main.py` with your dataset variables.

2. **EigenPro Configuration**
   - **Scalability**: EigenPro++ is capable of running on a single GPU with support for up to 1M centers.
   - **Kernel and Bandwidth**: Configure your kernel and bandwidth in the EigenPro settings. Use `kernel_fn = None` for automatic selection.
   - **Precision**: Opt for `torch.float32` for higher precision. `float16` is generally sufficient for most datasets.

3. **Usage**
   - Review the comments in `run_eigenpro` to select the best configuration for your use case. If you want your model centers to be the same as the whole data set(traditional kernel machine), then set `use_all_x_as_centers = 1` to switch to EigenPro2.0 method which is a faster method for this special case. intoduced in [[2-3]](#References)

4. **Execution**
   - To run the updated EigenPro, execute `python main.py` in your command line.

---

# References
1. Abedsoltan, Amirhesam and Belkin, Mikhail and Pandit, Parthe, “Toward Large Kernel Models,” Proceedings of the 40th International Conference on Machine Learning, ICML'23, JMLR.org, 2023. [Link](https://proceedings.mlr.press/v202/abedsoltan23a/abedsoltan23a.pdf)
2. Siyuan Ma, Mikhail Belkin, "Kernel machines that adapt to GPUs for effective large batch trainin", Proceedings of the 2nd SysMLConference,Palo Alto, CA,USA, 2019. [Link](https://mlsys.org/Conferences/2019/doc/2019/171.pdf)
3. Siyuan Ma, Mikhail Belkin, Diving into the shallows: a computational perspective on large-scale shallow learning, Advances in Neural Information Processing Systems 30 (NeurIPS 2017). [Link](https://proceedings.neurips.cc/paper_files/paper/2017/file/bf424cb7b0dea050a42b9739eb261a3a-Paper.pdf)
4. Giacomo Meanti, Luigi Carratino, Lorenzo Rosasco, Alessandro Rudi, “Kernel methods through the roof: handling billions of points efficiently,” Advances in Neural Information Processing Systems, 2020. [Link](https://proceedings.neurips.cc/paper_files/paper/2020/file/a59afb1b7d82ec353921a55c579ee26d-Paper.pdf)
5. Alessandro Rudi, Luigi Carratino, Lorenzo Rosasco, “FALKON: An optimal large scale kernel method,” Advances in Neural Information Processing Systems, 2017. [Link](https://papers.nips.cc/paper_files/paper/2017/file/05546b0e38ab9175cd905eebcc6ebb76-Paper.pdf)
6. Ulysse Marteau-Ferey, Francis Bach, Alessandro Rudi, “Globally Convergent Newton Methods for Ill-conditioned Generalized Self-concordant Losses,” Advances in Neural Information Processing Systems, 2019. [Link](https://arxiv.org/pdf/1907.01771.pdf)










