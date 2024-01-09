# Tutorial: How to Use This Repository

## Data Preparation
Replace `X`, `Y`, `X_val`, and `Y_val` in `main.py` with your dataset variables.

## EigenPro Configuration
- **Scalability**: The latest version is capable of running on a single GPU with support for up to 1M centers.
- **Kernel and Bandwidth**: Configure your kernel and bandwidth in the EigenPro settings. Use `kernel_fn = None` for automatic selection.
- **Precision**: Opt for `torch.float32` for higher precision. `float16` is generally sufficient for most datasets.

## Usage
Review the comments in `run_eigenpro` to select the best configuration for your use case.

## Execution
To run the updated EigenPro, execute `python main.py` in your command line.

---

# Benchmark Results

## Performance Highlights
- **Performance Boost**: The new version of EigenPro offers an estimated 100x speed increase from the previous iterations.
- **Comparison with FALKON**: It demonstrates considerable speed advantages over FALKON.

## Benchmarking Details
Benchmarks were conducted on the CIFAR5M dataset using extracted features from the pretrained 'mobilenet_2' network available in the timm library. The benchmarks processed the full 5 million samples of CIFAR5M for one epoch/iteration for all versions of EigenPro and FALKON.

![Performance Comparison Table](benchmark.png)


