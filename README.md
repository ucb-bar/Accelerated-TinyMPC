# Accelerated TinyMPC

This repository contains the implementation of **Accelerated TinyMPC**, a project focused on design space exploration of embedded architectures for real-time optimal control. It aims to profile, optimize, and accelerate  [TinyMPC](https://tinympc.org/) workloads using scalar CPUs, vector architectures, and domain-specific accelerators such as **RISC-V** cores, **Saturn vector units**, and **Gemmini systolic arrays**. The project provides profiling and acceleration results; and demonstrates quantitative performance and area trade-offs across various architectures, aiming to find the most effective hardware optimizations for robotic model-based control algorithms.


## Report

For detailed insights into the design space exploration and performance evaluations, please refer to the attached report:  
[Design-Space-Exploration-of-Embedded-SoC-Architectures-for-Real-Time-Optimal-Control.pdf](Design-Space-Exploration-of-Embedded-SoC-Architectures-for-Real-Time-Optimal-Control.pdf)


----

# Usage

## Building

1. On terminal, clone this repo

   ```bash
   git clone https://github.com/ucb-bar/Accelerated-TinyMPC.git
   cd Accelerated-TinyMPC


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

* Run the `quadrotor_hovering` example

```bash
./examples/quadrotor_hovering
```

* Run the `codegen_cartpole` example then follow the same building steps inside that directory

```bash
./examples/codegen_cartpole
```

## Citing TinyMPC

If you are using TinyMPC, we encourage you to

* [Cite the related papers](https://tinympc.org/docs/citing/),
* Put a star on this repository.

**Excited to hear your TinyMPC success stories—share them with us!**

## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/TinyMPC/TinyMPC/issues). All types of issues are welcome including bug reports, documentation typos, feature requests and so on.

## Running on MCUs

Numerical benchmarks against other solvers on MCUs are available [here](https://github.com/RoboticExplorationLab/mcu-solver-benchmarks).

## Notes
