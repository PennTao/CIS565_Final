
#include <thrust/scan.h>
extern "C"
{
void ExclusiveScan(int * vertices_number, int n)
{
	thrust::exclusive_scan(thrust::device_ptr<int>(vertices_number), 
                           thrust::device_ptr<int>(vertices_number + n),
                           thrust::device_ptr<int>(vertices_number));

}
}