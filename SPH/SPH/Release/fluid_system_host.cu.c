#line 1 "C:/Users/Yuanhui/AppData/Local/Temp/tmpxft_00001bbc_00000000-0_fluid_system_host.cudafe1.gpu"
#line 32 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
struct FluidParams;
#line 478 "C:\\Program Files\\Microsoft Visual Studio 9.0\\VC\\include\\crtdefs.h"
typedef unsigned size_t;
#include "crt/host_runtime.h"
#line 93 "C:\\Program Files\\Microsoft Visual Studio 9.0\\VC\\include\\time.h"
typedef long clock_t;
#line 28 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\cutil_math.h"
typedef unsigned uint;
#line 32 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
struct FluidParams {
#line 33 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int numThreads;
#line 33 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int numBlocks;
#line 34 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int gridThreads;
#line 34 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int gridBlocks;
#line 35 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int voxelThreads;
#line 35 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int voxelBlocks;
#line 36 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int szPnts;
#line 36 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int szHash;
#line 36 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int szGrid;
#line 37 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int stride;
#line 37 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int pnts;
#line 37 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int cells;
#line 38 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int chk;
#line 39 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int xDim;
#line 39 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int yDim;
#line 39 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
int zDim;
#line 40 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float smooth_rad;
#line 40 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float r2;
#line 40 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float sim_scale;
#line 40 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float visc;
#line 41 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
struct float3 min;
#line 41 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
struct float3 max;
#line 41 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
struct float3 res;
#line 41 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
struct float3 size;
#line 41 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
struct float3 delta;
#line 42 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float speedLimit;
#line 43 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float pdist;
#line 43 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float pmass;
#line 43 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float rest_dens;
#line 43 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float stiffness;
#line 44 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float poly6kern;
#line 44 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float spikykern;
#line 44 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float lapkern;
#line 45 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float slope;
#line 46 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float damp;
#line 47 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float particleRadius;
#line 48 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float xminSin;
#line 48 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float xmaxSin;
#line 49 "c:\\users\\yuanhui\\desktop\\sph\\fluids\\fluid_system_host.cuh"
float gravity;};
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);

#include "Release/fluid_system_host.cu.stub.c"
