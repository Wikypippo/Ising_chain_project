# Transverse-Field Ising Chain Simulation

This repository contains a High-Performance Computing (HPC) implementation of the one-dimensional Transverse-Field Ising Model (TFIM). The project focuses on computing the **ground-state energy** and the **squared magnetization** ($\langle M_z^2 \rangle$) in order to investigate the quantum phase transition (QPT) at zero temperature.

## Project Overview

The exact simulation of quantum many-body systems is fundamentally limited by the exponential growth of the Hilbert space, whose dimension scales as $2^N$, where $N$ is the number of spins.

This project investigates the computational limits of exact diagonalization by progressively optimizing the simulation architecture to overcome hardware bottlenecks, particularly the **memory wall**. The ground state is obtained through **Imaginary Time Evolution (ITE)**.

## Key Optimizations

### Matrix-Free Hamiltonian

A dense Hamiltonian representation requires storing a $2^N \times 2^N$ matrix, quickly exhausting both memory capacity and memory bandwidth.

The final implementation computes the action of the Hamiltonian on the state vector **on-the-fly** using bitwise operations, reducing the memory complexity from

$$
\mathcal{O}(2^{2N})
$$

to

$$
\mathcal{O}(2^N),
$$

allowing the working set to remain almost entirely within the CPU caches.

### Hybrid Parallelization (MPI + OpenMP)

The computation is parallelized at two levels:

- **MPI** distributes independent simulations across different values of the transverse field $h$.
- **OpenMP** parallelizes the matrix-vector operations within each MPI process.

This hybrid approach efficiently exploits both distributed-memory and shared-memory architectures.

### Optimized Observable Evaluation

Instead of computing the root-mean-square (RMS) magnetization, the implementation directly evaluates the squared magnetization,

$$
\langle M_z^2 \rangle,
$$

avoiding unnecessary `sqrt()` calls inside the computationally intensive loops.

---

# Repository Structure

| File | Description |
|------|-------------|
| `main_seriale.c` | Baseline serial implementation using a dense Hamiltonian matrix. |
| `pmain.c` | Parallel dense-matrix implementation (MPI/OpenMP), mainly used to illustrate cache-thrashing effects. |
| `pmain_sparse.c` | Final optimized Matrix-Free implementation with hybrid MPI + OpenMP parallelization. |

---

# Compilation and Execution

The project is written in **C** and requires:

- GCC
- OpenMP
- OpenMPI (only for the hybrid version)

## 1. Serial Version (Dense Matrix)

Suitable for small systems ($N \lesssim 10$).

```bash
gcc -O3 main_seriale.c -o main -lm
```

Run:

```bash
./main 10
```

---

## 2. Shared-Memory Version (OpenMP)

Optimized Matrix-Free implementation for multicore CPUs.

Compile:

```bash
gcc -O3 -fopenmp pmain_sparse.c -o smain1 -lm
```

Run on four threads:

```bash
export OMP_NUM_THREADS=4
./smain1 15
```

---

## 3. Hybrid MPI + OpenMP Version

Designed for distributed-memory HPC clusters.

Compile:

```bash
mpicc -O3 -fopenmp -DUSE_MPI pmain_sparse.c -o smain -lm
```

Run using two MPI processes and two OpenMP threads per process:

```bash
export OMP_NUM_THREADS=2
mpirun --mca btl tcp,self --oversubscribe -np 2 ./smain 10
```

---

# Performance Profiling

The implementation was profiled using the Linux `perf` tool to evaluate CPU performance, including:

- Instructions per Cycle (IPC)
- Cache references
- Cache misses
- Overall execution time

Example:

```bash
perf stat \
-e task-clock,cycles,instructions,cache-references,cache-misses \
./smain1 15
```

---

# Program Output

The simulation produces CSV-formatted output with the following columns:

```text
h,Energy,Mz2
```

where:

- **h** is the transverse magnetic field,
- **Energy** is the estimated ground-state energy,
- **Mz2** is the expectation value $\langle M_z^2 \rangle$.

The transverse field is sampled in the interval

$$
0 \le h \le 1,
$$

approaching the critical point

$$
h_c = 1.
$$

---

# Main Features

- Exact simulation of the 1D Transverse-Field Ising Model
- Imaginary Time Evolution ground-state solver
- Matrix-Free Hamiltonian implementation
- Hybrid MPI + OpenMP parallelization
- Cache-friendly memory layout
- Optimized observable computation
- Performance profiling with Linux `perf`