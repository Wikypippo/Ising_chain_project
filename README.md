# Transverse-Field Ising Chain Simulation

This repository contains a High-Performance Computing (HPC) implementation of the one-dimensional Transverse-Field Ising Model (TFIM).

The project focuses on computing the **ground-state energy** and the **squared magnetization** ($\langle M_z^2 \rangle$) in order to study the quantum phase transition (QPT) at zero temperature.

---

## Project Overview

The exact simulation of quantum many-body systems is fundamentally limited by the exponential growth of the Hilbert space, whose dimension scales as $2^N$, where $N$ is the number of spins.

This project explores the computational limits of exact diagonalization and imaginary time evolution by progressively optimizing the implementation to overcome key HPC bottlenecks, particularly the **memory wall**.

The ground state is computed using **Imaginary Time Evolution (ITE)**, and the code evolves through multiple levels of optimization, from a dense-matrix serial approach to a fully hybrid MPI + OpenMP matrix-free solver.

---

# Compilation and Execution

The repository contains three main implementations:

- `main_seriale.c` → serial dense-matrix baseline
- `main_parallelo.c` → OpenMP parallel dense version with optional MPI support
- `main_matrixfree.c` → optimized matrix-free version with optional MPI support

MPI is enabled via the `-DUSE_MPI` compilation flag.

---

## 1. Serial Dense Matrix Version

Baseline implementation using an explicit $2^N \times 2^N$ Hamiltonian matrix.  
Used for validation and to highlight memory scaling limitations.

### Compile

```bash
gcc -O3 -fopenmp main_seriale.c -o main_seriale -lm
```

### Run

```bash
./main_seriale 10
```

---

## 2. OpenMP Parallel Version (No MPI)

Shared-memory parallel implementation using OpenMP.

Depending on compilation, this version still use a dense representation.

### Compile

```bash
gcc -O3 -fopenmp main_parallelo.c -o main_parallelo -lm
```

### Run

```bash
./main_parallelo 10
```

---

## 3. MPI + OpenMP Parallel Version

Hybrid distributed-memory + shared-memory implementation.

MPI distributes independent simulations over different transverse field values $h$, while OpenMP parallelizes inner computations.

### Compile

```bash
mpicc -O3 -fopenmp -DUSE_MPI main_parallelo.c -o main_parallelo_mpi -lm
```

### Run

```bash
mpirun --oversubscribe -np 2 ./main_parallelo_mpi 10
```

---

## 4. Matrix-Free OpenMP Version (No MPI)

Fully optimized implementation using a matrix-free Hamiltonian.

The Hamiltonian action is computed on-the-fly using bitwise operations, removing the need to store the full $2^N \times 2^N$ matrix.

### Compile

```bash
gcc -O3 -fopenmp main_matrixfree.c -o main_matrixfree_omp -lm -lpapi
```

### Run

```bash
./main_matrixfree_omp 10
```

---

## 5. Matrix-Free MPI + OpenMP Version (Final Optimized)

Fully optimized hybrid implementation:

- MPI distributes simulations over different values of the transverse field $h$
- OpenMP parallelizes the matrix-free Hamiltonian-vector product
- Memory footprint reduced from exponential matrix storage to $\mathcal{O}(2^N)$ state vectors only

MPI is enabled via `-DUSE_MPI`.

### Compile

```bash
mpicc -O3 -fopenmp -DUSE_MPI main_matrixfree.c -o main_matrixfree_mpi -lm -lpapi
```

### Run

```bash
mpirun --oversubscribe -np 2 ./main_matrixfree_mpi 10
```

---

# Key Features

- Exact simulation of the 1D Transverse-Field Ising Model
- Imaginary Time Evolution ground-state solver
- Progressive optimization from dense to matrix-free formulations
- OpenMP parallelization for shared-memory systems
- MPI + OpenMP hybrid scaling for distributed systems
- Cache-friendly and memory-efficient implementation
- Performance profiling support via `perf`