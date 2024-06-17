# Experimental Results

### Matrix Multiplication
Hey, we've gotta start somewhere. I'm sure that this has been done before, but it makes a good test run to make sure I've learned how to use the benchmarking & optimization tools correctly.

Using the script `time_trials/test_matfuncs.py` I found that for matrices of size ~175x175 and larger, cupy matmult is always faster. Below that size, numpy starts to win out. Naturally, there is some overhead in communicating the data in the matrices back and forth to the GPU which accounts for the bulk of the cupy execution time represented in these plots. Notably, cupy has a pretty significant initialization time, so all benchmarks will need to use the `warmup_rounds` flag to be truly accurate.
![150x150](../blob/benchmark_matmul_150.svg)
![175x175](../blob/benchmark_matmul_175.svg)

I wonder how this number changes with other data types (int/int8, binary, ternary) and operations. 