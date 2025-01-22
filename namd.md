# NAMD 水分子动力学 Benchmark 分析

## Molecules Dynamic Background

### 什么是分子模拟

![Alt text|center](assets/namd/image-0.png)

### 分子模拟建模尺度

![Alt text|center](assets/namd/image-1.png)

### 为何需要分子动力学计算加速

![Alt text|center](assets/namd/image-2.png)

### 计算核心难点 — 力场计算

![Alt text|center](assets/namd/image-3.png)

## Benchmark Brief

![Alt text|center](assets/namd/image-4.png)

![Alt text|center](assets/namd/image-5.png)

## Workload Analysis

### Compute Work Overview

![Alt text|center](assets/namd/image-6.png)

### GPU Compute Work

#### Overview

![Alt text|center](assets/namd/image-7.png)

#### Patch Decomposition

![Alt text|center](assets/namd/image-8.png)

#### Computes and Tiles

![Alt text|center](assets/namd/image-9.png)

#### Non-bonded Force Calculation

![Alt text|center](assets/namd/image-10.png)

#### Warp load balance with neighbor list sorting

![Alt text|center](assets/namd/image-11.png)

#### Inside non-bonded force computation kernels

![Alt text|center](assets/namd/image-12.png)

#### Summary

![Alt text|center](assets/namd/image-13.png)

### Top-Down Analysis

#### Hot Spot Analysis

- Function Hot spots

	![Alt text|center](assets/namd/image-14.png)
- Kernel Hot spots

	![Alt text|center](assets/namd/image-15.png)
- Kernel Implementation

	![Alt text|center](assets/namd/image-16.png)

### Bottom-Up Analysis

#### GPU Throughput

![Alt text|center](assets/namd/image-17.png)

#### Roofline Model

![Alt text|center](assets/namd/image-18.png)

#### Pipeline Utilization

![Alt text|center](assets/namd/image-19.png)

#### Memory Chart

![Alt text|center](assets/namd/image-20.png)

#### Share Memory Atomics

![Alt text|center](assets/namd/image-21.png)

### Cross Validation

![Alt text|center](assets/namd/image-22.png)

## Summary

![Alt text|center](assets/namd/image-23.png)

## What-If Analysis: FP32 As Accumulate Type

![Alt text|center](assets/namd/image-24.png)

![Alt text|center](assets/namd/image-25.png)

## References

1. [Improving NAMD Performance on Multi-GPU Platforms](https://charm.cs.illinois.edu/workshops/charmWorkshop2018/slides/CharmWorkshop2018_namd_hardy.pdf)
2. [NAMD 3.0alpha GPU benchmarking results](http://www.ks.uiuc.edu/Research/namd/benchmarks/)
3. [Delivering up to 9X the Throughput with NAMD v3 and NVIDIA A100 GPU](https://developer.nvidia.com/blog/delivering-up-to-9x-throughput-with-namd-v3-and-a100-gpu/)
4. [Pushing the limit of molecular dynamics with ab initio accuracy to 100 million atoms with machine learning](https://arxiv.org/pdf/2005.00223.pdf)
5. [86 PFLOPS Deep Potential Molecular Dynamics simulation of 100 million atoms with ab initio accuracy](https://arxiv.org/pdf/2004.11658.pdf)