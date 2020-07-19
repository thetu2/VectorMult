# VectorMult
## Build Instructions (Visual Studio 2017+)

- Open folder containing project files using VS
- Build->Build All
- Select startup item->emult_perf.exe


## Operating Instuctions
The point of this code is to test various datatypes including a half precision complex type. To select the datatype to test, set the `TYPE` macro on line 26.

Set the 'COMPCAP' macro to the the gpu's compute capability.

In Cmakelist.txt, set the line 'string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_XX,code=sm_XX")' to the compute capability of the gpu (e.g. -gencode arch=compute_61,code=sm_61 for gtx 1080)

## Results
The code was run using an NVIDIA GTX 1080 ti GPU. The number of trials was 10000 and the number of elements to multiply per trial was 50000.

Real: 0.128ms

Complex32: 0.214ms

2x half: 0.127ms

half2: 0.127ms (same for both methods)


