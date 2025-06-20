# CF-Merge: Bank conflict free GPU mergesort
GPU pairwise mergesort using a bank conflict free merging stage from "Eliminating Bank Conflicts in GPU Mergesort".  
Experiments on a NVIDIA RTX 2080 Ti show that CF-Merge eliminates the slowdowns due to bank conflicts.

[Throughput results (elements per microsecond) for Thrust and CF-Merge on a NVIDIA RTX 2080 Ti using the constructed worst-case inputs. Thrust results are in yellow and CF-Merge results are in blue.
The short dashed lines represent software parameters 15 elements per thread (E) and 512 threads per block (u); and the long dashed lines represent software parameters 17 elements per thread (E) and 256 threads per block (u). The x-axis is displayed on a logarithmic scale.](figures/rtx2080ti_sort_worst.pdf)

[Throughput results (elements per microsecond) for Thrust and CF-Merge on a NVIDIA RTX 2080 Ti using parameters 15 elements per thread (E) and 512 threads per block (u).
Thrust results are in yellow and CF-Merge results are in blue.
The dashed lines represent the constructed worst-case inputs and the dotted lines represent uniform random inputs.
The x-axis is displayed on a logarithmic scale.](figures/rtx2080ti_sort_15.pdf)

[Throughput results (elements per microsecond) for Thrust and CF-Merge on a NVIDIA RTX 2080 Ti using parameters 17 elements per thread (E)  and 256 threads per block (u).
Thrust results are in yellow and CF-Merge results are in blue.
The dashed lines represent the constructed worst-case inputs and the dotted lines represent uniform random inputs.
The x-axis is displayed on a logarithmic scale.](figures/rtx2080ti_sort_17.pdf)

Experimental setup:
* NVIDIA RTX 2080 Ti
* CUDA 11
* Thrust v1.9.9
* Worst-case inputs can be downloaded [here](https://drive.google.com/file/d/1_NlA1FIxeiKB1QPGh2iVKQKK3pocn43T/view?usp=sharing)

## Files
1. `test/sort_int_random.cu` - Test harness for random inputs  
Command line arguments:
   * Total number of warps (positive power of 2 required)
   * RNG seed value
3. `test/sort_int_worst.cu` - Test harness for constructed inputs  
Command line arguments:
   * Total number of warps (positive power of 2 required)
   * Path of the directory containing binary files for the worst-case constructed inputs
4. `sort.h` - Modified Thrust code using CF-Merge
5. `Makefile` - Makefile for compiling test harness programs

## Running CF-Merge
1. Overwrite the default `sort.h` file in Thrust located in `thrust-1.9.9/thrust/system/cuda/detail/`
```bash
cp sort.h thrust-1.9.9/thrust/system/cuda/detail/sort.h
```

2. Compile test harness programs
```bash
make
```

3. Run test harness programs
```bash
./sort_int_random_15.out <total number of warps (positive power of 2 required)> <RNG seed value>
./sort_int_worst_15.out <total number of warps (positive power of 2 required)> <directory filepath>
./sort_int_random_17.out <total number of warps (positive power of 2 required)> <RNG seed value>
./sort_int_worst_17.out <total number of warps (positive power of 2 required)> <directory filepath>
```
