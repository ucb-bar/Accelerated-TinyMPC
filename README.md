# Accelerated TinyMPC

This repository contains the implementation of **Accelerated TinyMPC**, a project focused on design space exploration of embedded architectures for real-time optimal control. It aims to profile, optimize, and accelerate  [TinyMPC](https://tinympc.org/) workloads using scalar CPUs, vector architectures, and domain-specific accelerators. The project provides profiling and acceleration results; and demonstrates quantitative performance and area trade-offs across various architectures, aiming to find the most effective hardware optimizations for robotic model-based control algorithms.

## Building on Ubuntu

1. On terminal, clone this repo

```bash
git clone git@github.com:TinyMPC/TinyMPC.git
```

2. Navigate to root directory and run

```bash
mkdir build && cd build
```

3. Run CMake configure step

```bash
cmake ../
```

4. Build TinyMPC

```bash
make 
```

## Examples

* Run the quadrotor hovering example

```bash
./examples/example_quadrotor_hovering
```

* Run the codegen example then follow the same building steps inside that directory

```bash
./examples/example_codegen
```

## Running on MCUs

To be documented

## Notes
