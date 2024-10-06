+++
title = 'Introduction to CUDA programming'
description = "A simple introduction to CUDA programming"
date = 2024-10-06T16:08:06+05:30
draft = false
tags = ["cuda", "gpu", "deep-learning"]
+++

## What is CUDA?

CUDA stands for `Compute Unified Device Architecture`, developed by **NVIDIA**.

{{< figure src="/01-cuda-introduction/nvidia-cuda.webp" title="Nvidia CUDA" >}}

Earlier, NVIDIA was known for its graphics processing units (GPUs), which were designed to handle tasks related to rendering graphics. However, the GPUs were not designed for general-purpose computing tasks, and NVIDIA's GPUs were primarily used for graphics rendering.

In 2014, NVIDIA released the first CUDA-enabled GPU, the **Tesla GPU**, which was designed to handle general-purpose computing tasks.

---

## Terminologies

- **`host`** refers to the **`CPU`** and its memory
- **`device`** refers to the **`GPU`** and its memory
- **`kernels`** are functions executed on the **device (GPU)**

---

## A typical sequence of a CUDA program

1. Declare and allocate host (`cpu`) and device (`gpu`) memory.
2. Initialize host data. (`cpu`)
3. Transfer data from the host to the device. (`cpu => gpu`)
4. Execute one or more kernels. (`functions on gpu`)
5. Transfer results from the device to the host. (`gpu => cpu`)

Keeping this sequence of operations in mind, we'll have a look at the code snippets soon.

---

## Some more info

Just like we use `malloc` and `free` to allocate and free memory, we use **`cudaMalloc`** and **`cudaFree`** to allocate and free memory on the GPU.

```c
int *d_a; // Pointer for device memory
cudaMalloc(&d_a, 100 * sizeof(int)); // Allocating space for 100 integers on the GPU
cudaFree(d_a); // Freeing the memory
```

To copy data from `host to device` or vice versa, we use **`cudaMemcpy`**.

```c
cudaMemcpy (void *dst, const void *src, size_t count, cudaMemcpyKind kind);
/*
dst: The destination memory address (either on the GPU or CPU).
src: The source memory address (either on the GPU or CPU).
count: The number of bytes to copy.
kind: direction of the copy, specified by the cudaMemcpyKind enum. Common options:
    cudaMemcpyHostToDevice: Copy data from host (CPU) to device (GPU).
    cudaMemcpyDeviceToHost: Copy data from device (GPU) to host (CPU).
    cudaMemcpyDeviceToDevice: Copy data between two locations on the device (GPU).
    cudaMemcpyHostToHost: Copy data between two locations on the host (CPU).
*/
```

To mark a function as a **`kernel`**, we use **`__global__`** and **`__device__`**:

- `__global__`: The function is a kernel, executed on the GPU, and called from the CPU.
- `__device__`: The function is executed on the GPU but can only be called from other device (GPU) functions.
- `__host__`: The function is executed on the CPU (this is the default if no keyword is specified).

---

## A simple CUDA program

The below code performs `a*x+y` on the GPU for 1M elements, and returns the result on the CPU. And, finally we compute the error and print it.

**SAXPY** stands for `Single-precision A*X Plus Y`, and is a good “hello world” example for parallel computation.

```c
// save this file: `saxpy.cu`
#include <stdio.h>

// this function is `kernel` and executed on the GPU, and called from the CPU
__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  // declare pointers for host and device memory
  float *x, *y, *d_x, *d_y;

  // allocate host memory
  x = (float*)malloc(N*sizeof(float)); 
  y = (float*)malloc(N*sizeof(float));

  // allocate device memory
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  } // initialize host data

  // transfer data from host to device
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  // call kernel function by specifying number of blocks and threads per block
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y); 

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
```

---

## Launching a `Kernel`

The saxpy kernel is launched by the statement:

```c
saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
```

The information between the **`triple chevrons`** is the execution configuration, which dictates how many device threads execute the kernel in parallel.

Here, we are launching the kernel `saxpy` on the GPU, with the number of blocks and threads per block specified. The number of threads per block is 256, and the number of blocks is the number of elements divided by 256 rounded up.

```diff
In the execution configuration
- The first argument specifies the `number of thread blocks` in the grid,
- and the second specifies the `number of threads` in a thread block.
```

- There's no limit on the number of thread blocks, but the number of threads in each block must be less than 1024. (or 512 for older GPUs)

Thread blocks and grids can be made one-, two- or three-dimensional by passing dim3 (a simple struct defined by CUDA with x, y, and z members) values for these arguments, but for this simple example we only need one dimension so we pass integers instead. 

Otherwise, we could've also done:

```c
saxpy<<<dim3(N/256, 2, 1), dim3(256, 1, 1)>>>(N, 2.0f, d_x, d_y);
```

And, we can use `threadIdx.x, threadIdx.y, threadIdx.z` to get the thread id in each dimension. Similarly, for `blockIdx` & `blockDim`.

For more details on [threads, blocks, and grids, refer to this video](https://www.youtube.com/watch?v=kzXjRFL-gjo).

For a gpu thread, to get to know which index it should process, we use this very often:

```c
// these variables are available by default
// get the block id, block size, and thread id
int i = blockIdx.x*blockDim.x + threadIdx.x; 
```

---

## Running the program

To run the program:

```bash
nvcc -o saxpy saxpy.cu # nvcc - nvidia cuda compiler
./saxpy
```

---

## That's it!

We've covered the basics of CUDA programming. We've learned how to allocate memory on the GPU, transfer data between the host and the device, and launch a kernel on the GPU.

Thanks for reading this far. Reach out to me on [Twitter](https://twitter.com/deependu__) or [GitHub](https://github.com/deependujha) if you have any questions or suggestions.

---

## References

- [An easy introduction to CUDA C & C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [NVIDIA CUDA: Threads, Thread Blocks and Grids](https://www.youtube.com/watch?v=kzXjRFL-gjo)

---
