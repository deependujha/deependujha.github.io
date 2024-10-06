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

