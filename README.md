# cuda-kernel-profiling

This repository contains CUDA kernel examples for learning GPU programming and profiling.  
The first example is `vectorAdd.cu`, which performs vector addition on the GPU.

## How to Run on Google Colab

1. Open Google Colab: https://colab.research.google.com/
2. Go to **Runtime → Change runtime type → Hardware accelerator: GPU**
3. In a new cell, run:
    ```bash
    !nvidia-smi
    !nvcc --version
    ```
4. Clone this repository:
    ```bash
    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    %cd YOUR_REPO
    ```

5. !nvcc -arch=sm_75 -gencode arch=compute_75,code=compute_75 vectorAdd.cu -o vectorAdd

6. Compile and run:
    ```bash
    !./vectorAdd
    ```

