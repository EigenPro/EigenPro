## Tutorial: How to Use This Repository

1. **Data Preparation**: 
   In `main.py`, replace `X`, `Y`, `X_val`, and `Y_val` with your dataset.

2. **Eigenpro Configuration**:
   - **Kernel and Bandwidth**: Choose your preferred kernel and bandwidth in the Eigenpro configs section. If unsure, set `kernel_fn = None` and we'll auto-select for you.
   - **Precision**: Change `type` to `torch.float32` for higher precision. `float16` is generally adequate for most datasets.

3. **Usage**: 
   Review the comments in `run_eigenpro` to determine the best usage for your situation.

4. **Execution**: 
   After editing, run the script with `python main.py`.
