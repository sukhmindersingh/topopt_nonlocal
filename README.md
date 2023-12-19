# TopOpt Nonlocal

This is a density-based topology optimization code based on [deal.II](https://github.com/dealii/dealii) finite element library and MMA (Method of Moving Asymptotes) optimizer from [ParOpt](https://smdogroup.github.io/paropt/) library

## Features

- Preconditioned conjugate gradient, matrix-free solver for linear elasticity
- MPI parallelization
- 2D and 3D problems

## How to run

- Install [deal.II](https://github.com/dealii/dealii) finite-element library using [SPACK](https://github.com/spack/spack)

```bash
spack install dealii@9.5.1+optflags
```

- Install [ParOpt](https://smdogroup.github.io/paropt/) library for MMA (Method of Moving Asymptotes) solver

- Compile the code, using `-DPLANAR=1` for running two-dimensional problems

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPLANAR=1 -DCMAKE_CXX_FLAGS="-O3 -fopenmp -w -march=native" ../
```

- Run the code, e.g., using 4 MPI processes and 2 OpenMP threads per process

```bash
export OMP_NUM_THREADS=2
export DEAL_II_NUM_THREADS=2
mpirun -np 4 ./topopt input.yaml
```
