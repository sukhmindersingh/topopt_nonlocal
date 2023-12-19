# TopOpt Nonlocal

This is a finite element code based on deal.II finite element library and MMA optimizer from ParOpt library

S. Singh, L. Pflug, F. Wein and M. Stingl, A nonlocal approach to graded surface modeling in topology optimization.

## Features:

- Preconditioned conjugate gradient, matrix-free solver for linear elasticity
- MPI parallelization
- 2D and 3D problems

## How to run:

- Install deal.II library using SPACK

```bash
spack install dealii@9.5.1+optflags
```

- Install [ParOpt](https://smdogroup.github.io/paropt/) library

- Compile the code

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPLANAR=1 -DCMAKE_CXX_FLAGS="-O3 -fopenmp -w -march=native" ../
```
