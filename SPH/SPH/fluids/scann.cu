#include <thrust/functional.h>
#include <thrust/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/host_vector.h>

extern "C"
{
void ExclusiveScan(int * vertices_number, int n)
{
	thrust::exclusive_scan(thrust::device_ptr<int>(vertices_number), 
                           thrust::device_ptr<int>(vertices_number + n),
                           thrust::device_ptr<int>(vertices_number));

}}