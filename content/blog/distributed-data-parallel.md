---
title: Distributed Data Parallel (DDP)
description: "Distributed Data Parallel (DDP) is a technique used to accelerate the training of machine learning models by simulatenously training the model across multiple GPUs or nodes, ensuring efficient utilization of resources and faster convergence."
date: 2025-06-26
authors:
  - name: Deependu
    link: https://github.com/deependujha
    image: https://github.com/deependujha.png
tags:
  - PyTorch
  - Distributed Training
  - Deep Learning
math: true
weight: 98
---

## 🧠 Concept: Training Workflows

#### **Single-GPU (No Parallelism)**

> DataLoader → batch → model → loss → backward → optimizer.step()

- Everything runs in one process, on one device.

---

#### **DDP**

> Multiple processes — each with its **own model replica and DataLoader**, running on a **separate GPU**.

![DDP](/blogs/distributed-training/ddp.png)

Each process does the following:

1. **Loads a different subset of data** using its own DataLoader (thanks to `DistributedSampler`).
2. **Performs forward + backward pass independently** on its own model replica.
3. During `backward()`, **gradients are synchronized across all processes** using `all-reduce`.
4. Each process computes the **average gradient**, ensuring consistent updates.
5. **Each optimizer step updates its local model**, but all replicas remain **in sync** because gradients were averaged.

✅ This gives you **data parallelism** with minimal communication overhead and no parameter divergence.

---

## `DDP` over `DataParallel (DP)`

> DataParallel is an **older approach** to data parallelism.
> DP is trivially simple (with just one extra line of code) but it is much less performant.

`DataParallel` | `DistributedDataParallel`
|--- | ---|
| More overhead; model is replicated and destroyed at each forward pass | Model is replicated only once |
|Only supports single-node parallelism | Supports scaling to multiple machines |
| Slower; uses multithreading on a single process and runs into Global Interpreter Lock (GIL) contention | Faster (no GIL contention) because it uses multiprocessing |



