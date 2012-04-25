/*
  FLUIDS v.1 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2008. Rama Hoetzlein, http://www.rchoetzlein.com

  ZLib license
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/


#include <cutil.h>	
//#include <stdio.h>
//#include <stdlib.h>
#include <string.h>
#include "table.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <thrust/scan.h>



#if defined(__APPLE__) || defined(MACOSX)
	#include <GLUT/glut.h>
#else
	#include <GL/glut.h>
#endif
#include <cuda_gl_interop.h>

//#include "radixsort.cu"

#include "fluid_system_kern.cu"			// build kernel

//#include <stdlib.h>


//#include "scan.cuh"
FluidParams					fcuda;

__device__ char*			bufPnts;		// point data (array of Fluid structs)
 float ** out_matrix;
float ** out_matrix_;
float3 * mean_point;
int * case_number;
 int * vertices_number;
 int * d_numVertsTable;
int ** d_edgeVertices;
int ** d_triTable;

int ** d_edgeVertices_;
int ** d_triTable_;


float3 * pointsList;
float3 * normalList;
bool flag=false;
float * out_density;
#define voxel_size 0.75
#define isolevel 1500
 

int max_triangle_number=80;
#define points_number 2047	

extern "C"
{
// Initialize CUDA

void cudaInit(int argc, char **argv)
{   
    //CUT_DEVICE_INIT(argc, argv);
 
	//cudaDeviceProp p;
	//cudaGetDeviceProperties ( &p, 0);
	
	/*printf ( "-- CUDA --\n" );
	printf ( "Name:       %s\n", p.name );
	printf ( "Revision:   %d.%d\n", p.major, p.minor );
	printf ( "Global Mem: %d\n", p.totalGlobalMem );
	printf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	printf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	printf ( "Warp Size:  %d\n", p.warpSize );
	printf ( "Mem Pitch:  %d\n", p.memPitch );
	printf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	printf ( "Const Mem:  %d\n", p.totalConstMem );*/
	
	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufPnts, 10 ) );
	flag=true;
	CUDA_SAFE_CALL(cudaMalloc ( (void**) &(mean_point), 1*sizeof(float3)));
/*	out_matrix_=new float *[1];
    for(int i=0;i<1;i++)
		{
			CUDA_SAFE_CALL(cudaMalloc((void **)& (out_matrix_[i]), 1*sizeof(float)));
		}
	CUDA_SAFE_CALL(cudaMalloc ( (void**) &(out_matrix), 1*sizeof(float *)));
	CUDA_SAFE_CALL( cudaMemcpy(out_matrix, out_matrix_, 1*sizeof(float *), cudaMemcpyHostToDevice));*/
	
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(out_density), 1*sizeof(float)));


////marching cube
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(case_number), 1*sizeof(int)));
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(vertices_number), 1*sizeof(int)));
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(d_numVertsTable), 1*sizeof(int)));


d_triTable_= new int *[1];
CUDA_SAFE_CALL(cudaMalloc((void **)& (d_triTable_[0]), 1*sizeof(int)));
d_edgeVertices_= new int *[1];
CUDA_SAFE_CALL(cudaMalloc((void **)& (d_edgeVertices_[0]), 1*sizeof(int)));
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(pointsList), sizeof(float3)));
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(normalList), sizeof(float3)));
	
//	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufPntSort, 10 ) );
//	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufHash, 10 ) );	
//	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufGrid, 10 ) );	
};
	
// Compute number of blocks to create
int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int maxThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( maxThreads, numPnts );
    numBlocks = iDivUp ( numPnts, numThreads );
}




void FluidClearCUDA ()
{
	CUDA_SAFE_CALL ( cudaFree ( bufPnts ) );
	if(flag)
	{
	//	CUDA_SAFE_CALL(cudaFree ( out_matrix_[0]));
        CUDA_SAFE_CALL(cudaFree ( d_triTable_[0]));
        CUDA_SAFE_CALL(cudaFree ( d_edgeVertices_[0]));

		flag=false;
	}
	else
	{
	/*	   for(int i=0;i<points_number;i++)
		{
			CUDA_SAFE_CALL(cudaFree ( out_matrix_[i]));
		}
        */

		   for(int i=0;i<256;i++)
		{
			CUDA_SAFE_CALL(cudaFree ( d_triTable_[i]));
			
		}

		   	   for(int i=0;i<12;i++)
		{
			CUDA_SAFE_CALL(cudaFree ( d_edgeVertices_[i]));
			
		}
	    
	}
   CUDA_SAFE_CALL ( cudaFree ( mean_point) );
  // 	CUDA_SAFE_CALL ( cudaFree ( out_density ) );
   CUDA_SAFE_CALL ( cudaFree ( case_number ) );
   CUDA_SAFE_CALL ( cudaFree ( vertices_number ) );
  CUDA_SAFE_CALL ( cudaFree ( d_numVertsTable) );
    delete d_edgeVertices_;
    delete d_triTable_;
	//delete out_matrix_;
	//CUDA_SAFE_CALL ( cudaFree ( out_matrix ) );
    CUDA_SAFE_CALL ( cudaFree ( d_edgeVertices ) );
    CUDA_SAFE_CALL ( cudaFree ( d_triTable ) );
}


void FluidSetupCUDA ( int num, int stride, float3 min, float3 max, float3 res, float3 size, int chk)
{	
	fcuda.min = make_float3(min.x, min.y, min.z);
	fcuda.max = make_float3(max.x, max.y, max.z);
	fcuda.res = make_float3(res.x, res.y, res.z);
	fcuda.size = make_float3(size.x, size.y, size.z);	
	fcuda.pnts = num;
	fcuda.delta.x = res.x / size.x;
	fcuda.delta.y = res.y / size.y;
	fcuda.delta.z = res.z / size.z;
	fcuda.cells = res.x*res.y*res.z;
	fcuda.chk = chk;

    float3 temp=max-min;
    
    fcuda.xDim=temp.x/voxel_size+1;
    fcuda.yDim=temp.y/voxel_size+1;
    fcuda.zDim=temp.z/voxel_size+1;
   
   
    computeNumBlocks ( fcuda.pnts, 512, fcuda.numBlocks, fcuda.numThreads);			// particles
    computeNumBlocks ( fcuda.cells, 512, fcuda.gridBlocks, fcuda.gridThreads);		// grid cell
    computeNumBlocks ( fcuda.xDim*fcuda.yDim*fcuda.zDim, 512, fcuda.voxelBlocks, fcuda.voxelThreads);
    
    
    fcuda.szPnts = (fcuda.numBlocks * fcuda.numThreads) * stride;        
    fcuda.szHash = (fcuda.numBlocks * fcuda.numThreads) * sizeof(uint2);		// <cell, particle> pairs
    fcuda.szGrid = (fcuda.gridBlocks * fcuda.gridThreads) * sizeof(uint);    
    fcuda.stride = stride;
    
    
//    printf ( "pnts: %d, t:%dx%d=%d, bufPnts:%d, bufHash:%d\n", fcuda.pnts, fcuda.numBlocks, fcuda.numThreads, fcuda.numBlocks*fcuda.numThreads, fcuda.szPnts, fcuda.szHash );
//    printf ( "grds: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", fcuda.cells, fcuda.gridBlocks, fcuda.gridThreads, fcuda.gridBlocks*fcuda.gridThreads, fcuda.szGrid, (int) fcuda.res.x, (int) fcuda.res.y, (int) fcuda.res.z );	

	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufPnts, fcuda.szPnts ) );	
	
	
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(mean_point),points_number *sizeof(float3)));
	
/*	out_matrix_=new float *[points_number];
    for(int i=0;i<points_number;i++)
		{
			CUDA_SAFE_CALL(cudaMalloc((void **)& (out_matrix_[i]), 9*sizeof(float)));
		}
	CUDA_SAFE_CALL(cudaMalloc ( (void**) &(out_matrix), points_number*sizeof(float *)));
	CUDA_SAFE_CALL( cudaMemcpy(out_matrix, out_matrix_, points_number*sizeof(float *), cudaMemcpyHostToDevice));*/
	
	int dim=fcuda.xDim*fcuda.yDim*fcuda.zDim;
	CUDA_SAFE_CALL(cudaMalloc ( (void**) &out_density,dim*sizeof(float)));


	////////////////////marching cube


CUDA_SAFE_CALL(cudaMalloc ( (void**) &(case_number), dim*sizeof(int))); 
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(vertices_number), (dim+1)*sizeof(int)));
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(d_numVertsTable), dim*sizeof(int)));
cudaMemcpy(d_numVertsTable, numVertsTable,256*sizeof(int),cudaMemcpyHostToDevice);


d_triTable_= new int *[256];
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(d_triTable), points_number*sizeof(int *)));

for(int i=0;i<256;i++)
{
CUDA_SAFE_CALL(cudaMalloc((void **)& (d_triTable_[i]), dim*sizeof(int)));
cudaMemcpy(d_triTable_[i],triTable[i],16*sizeof(int),cudaMemcpyHostToDevice);

}
cudaMemcpy(d_triTable,d_triTable_,256*sizeof(int *),cudaMemcpyHostToDevice);

d_edgeVertices_= new int *[12];
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(d_edgeVertices), 12*sizeof(int *)));
for(int i=0;i<12;i++)
{
  CUDA_SAFE_CALL(cudaMalloc((void **)& (d_edgeVertices_[i]), dim*sizeof(int)));
  cudaMemcpy(d_edgeVertices_[i],edgeVertices[i],2*sizeof(int),cudaMemcpyHostToDevice);
}

cudaMemcpy(d_edgeVertices,d_edgeVertices_,12*sizeof(int *),cudaMemcpyHostToDevice);

CUDA_SAFE_CALL(cudaMalloc ( (void**) &(pointsList), max_triangle_number*sizeof(float3)));
CUDA_SAFE_CALL(cudaMalloc ( (void**) &(normalList), max_triangle_number*sizeof(float3)));

//	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufPntSort, fcuda.szPnts ) );	
//	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufHash[0], fcuda.szHash ) );	
//	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufHash[1], fcuda.szHash ) );	
//	CUDA_SAFE_CALL ( cudaMalloc ( (void**) &bufGrid, fcuda.szGrid ) );
	
	printf ( "POINTERS\n");
	printf ( "bufPnts:    %p\n", bufPnts );
//	printf ( "bufPntSort: %p\n", bufPntSort );
//	printf ( "bufHash0:   %p\n", bufHash[0] );
//	printf ( "bufHash1:   %p\n", bufHash[1] );
//	printf ( "bufGrid:    %p\n", bufGrid );
	
	CUDA_SAFE_CALL ( cudaMemcpyToSymbol ( simData, &fcuda, sizeof(FluidParams) ) );
	cudaThreadSynchronize ();
}

void FluidParamCUDA ( float sim_scale, float smooth_rad, float mass, float rest, float stiff, float visc, float speedlimit, float slope, float damp, float pRadius, float xminF, float xmaxF, float gravity )
{
	fcuda.sim_scale = sim_scale;
	fcuda.smooth_rad = smooth_rad;
	fcuda.r2 = smooth_rad * smooth_rad;
	fcuda.pmass = mass;
	fcuda.rest_dens = rest;	
	fcuda.stiffness = stiff;
	fcuda.visc = visc;
	fcuda.speedLimit = speedlimit;
	fcuda.slope = slope;
	fcuda.damp = damp;
	fcuda.particleRadius = pRadius;
	fcuda.xminSin = xminF;
	fcuda.xmaxSin = xmaxF;
	fcuda.gravity = gravity;


	fcuda.pdist = pow ( fcuda.pmass / fcuda.rest_dens, 1/3.0f );
	fcuda.poly6kern = 315.0f / (64.0f * 3.141592 * pow( smooth_rad, 9.0f) );  //tao kernel functions
	fcuda.spikykern = -45.0f / (3.141592 * pow( smooth_rad, 6.0f) );
	fcuda.lapkern = 45.0f / (3.141592 * pow( smooth_rad, 6.0f) );	
	fcuda.surface_kernel=1.0f/(70*pow( smooth_rad, 6.0f));
	CUDA_SAFE_CALL( cudaMemcpyToSymbol ( simData, &fcuda, sizeof(FluidParams) ) ); //tao give value to simData
	cudaThreadSynchronize ();
}


void allocateCUDAspace(float *** out_matrix, float3 ** out_mean_p)
{
      
		/*float ** out_matrix_=new float *[fcuda.pnts];
		for(int i=0;i<fcuda.pnts;i++)
		{
			cudaMalloc((void **)& out_matrix_[i], fcuda.pnts*3*sizeof(float));
		}
		cudaMalloc((void **)out_matrix, fcuda.pnts*sizeof(float *));
		cudaMemcpy((*out_matrix), out_matrix_, fcuda.pnts*sizeof(float *), cudaMemcpyHostToDevice);

		cudaMalloc((void **)out_mean_p, fcuda.pnts*sizeof(float3));*/
}

void freeCUDAspace(float ** out_matrix, float3 *out_mean_p)
{
		/*for(int i=0;i<fcuda.pnts;i++)
		{
			cudaFree(out_matrix_);
		}
        cudaFree(out_matrix);*/
}
void TransferToCUDA ( char* data,  int numPoints )
{
	CUDA_SAFE_CALL( cudaMemcpy ( bufPnts, data, numPoints * fcuda.stride, cudaMemcpyHostToDevice ) );
	cudaThreadSynchronize ();
}

void TransferFromCUDA ( char* data, int numPoints )
{
	CUDA_SAFE_CALL( cudaMemcpy ( data, bufPnts, numPoints * fcuda.stride, cudaMemcpyDeviceToHost ) );	
	cudaThreadSynchronize ();	
	
	
} 

void TransferFromCUDATriangles(float3 * data, int numTriangles)
{
    /*for(int i=0;i<numTriangles;i++)
	 {
		 data[i*3]=make_float3(i,i,i);
		 data[i*3+1]=make_float3(i+1,i,i);
         data[i*3+1]=make_float3(i,i,i+1);
	 }*/
    CUDA_SAFE_CALL( cudaMemcpy ( data, pointsList, numTriangles * sizeof(float3), cudaMemcpyDeviceToHost ) );	
    cudaThreadSynchronize ();
}


void SPH_ComputePressureCUDA ()
{
	computePressure<<< fcuda.numBlocks, fcuda.numThreads>>> ( bufPnts,fcuda.pnts );	
    CUT_CHECK_ERROR( "Kernel execution failed");
    cudaThreadSynchronize ();	
/*	computePressureFast<<< fcuda.numBlocks, fcuda.numThreads>>> ( bufPnts, bufGrid,  fcuda.pnts );	
    CUT_CHECK_ERROR( "Kernel execution failed");
    cudaThreadSynchronize ();	*/
}

/*void SPH_ComputeForceCUDA (float ** out_matrix, float3 * out_mean_p)
{
   
	computeForce<<< fcuda.numBlocks, fcuda.numThreads>>> ( bufPnts, fcuda.pnts,out_matrix, out_mean_p );	
    CUT_CHECK_ERROR( "Kernel execution failed");
    cudaThreadSynchronize ();	
	computeForceFast<<< fcuda.numBlocks, fcuda.numThreads>>> ( bufPnts, fcuda.pnts );	
    CUT_CHECK_ERROR( "Kernel execution failed");
    cudaThreadSynchronize ();	
}*/

int SPH_ExclusiveScan( )
{
	int dim=fcuda.xDim*fcuda.yDim*fcuda.zDim+1;


	thrust::exclusive_scan(thrust::device_ptr<int>(vertices_number), 
		thrust::device_ptr<int>(vertices_number + dim),
                           thrust::device_ptr<int>(vertices_number));



	int temp;
	cudaMemcpy(&temp,&vertices_number[dim-1], sizeof(int),cudaMemcpyDeviceToHost);

	/*int *tempp=new int[dim];
	cudaMemcpy(tempp,vertices_number, dim*sizeof(int),cudaMemcpyDeviceToHost);
	for(int i=0;i<dim;i++)
	{
		printf("%d ",tempp[i]);
	}
	printf("\n");*/
	return temp;

	

}

void SPH_MarchingCube( )
{
	
    marching_cube<<<fcuda.voxelBlocks, fcuda.voxelThreads>>>(pointsList, normalList,d_numVertsTable,d_edgeVertices,d_triTable,out_density, fcuda.xDim, fcuda.yDim, fcuda.zDim, voxel_size, isolevel, vertices_number,case_number );
	//float3 * temp=new float3[378];
	//cudaMemcpy(temp,pointsList, 100*sizeof(float3), cudaMemcpyDeviceToHost);

	/*for(int i=0;i<378;i++)
	{
		printf("%f %f %f \n", temp[i].x,temp[i].y,temp[i].z);
	}*/
}
 




void SPH_ComputeForceCUDA ()
{

    
   computeForce<<< fcuda.numBlocks, fcuda.numThreads>>> ( bufPnts, fcuda.pnts,out_matrix, mean_point);	
   
 /* float3 temp;
   
  
  for(int i=0;i<2047;i++)
   {
     cudaMemcpy(&temp, &mean_point[i], sizeof(float3), cudaMemcpyDeviceToHost);

	 if(i%100==0)
		 printf("/////////////////////////////////////////////////////////////////////////////////////////%\n");

	 printf ( "%f  %f %f",temp.x,temp.y,temp.z);
	 printf("%\n");
   }
   
   CUT_CHECK_ERROR( "Kernel execution failed");
   cudaThreadSynchronize ();*/
    

}

void SPH_ComputeDensity()
{
  computeDensity<<<fcuda.voxelBlocks, fcuda.voxelThreads>>>(out_density, out_matrix, mean_point, bufPnts, fcuda.pnts, fcuda.xDim, fcuda.yDim, fcuda.zDim, voxel_size);

 	 int dim=fcuda.xDim*fcuda.yDim*fcuda.zDim;


	/*float * temp=new float[dim];
	  cudaMemcpy(temp, out_density, dim*sizeof(float),cudaMemcpyDeviceToHost);
	  
	  for(int i=0;i<dim;i++)
	  {
		  printf("%f ",temp[i]);
	  }*/
	  

}


void SPH_ComputeVertexNumber()
{
      compute_vertices_number<<<fcuda.voxelBlocks, fcuda.voxelThreads>>>(case_number,vertices_number, out_density, d_numVertsTable,fcuda.xDim, fcuda.yDim, fcuda.zDim, isolevel);
	// int dim=fcuda.xDim*fcuda.yDim*fcuda.zDim;

	/* int * temp=new int[dim];

	cudaMemcpy(temp, vertices_number, dim*sizeof(int),cudaMemcpyDeviceToHost);


	 for(int i=0;i<dim;i++)
	  {
		  printf("%d ",temp[i]);
	  }
	 	printf("\n");

	 cudaMemcpy(temp, case_number, dim*sizeof(int),cudaMemcpyDeviceToHost);

	  for(int i=0;i<dim;i++)
	  {
		  printf("%d ",temp[i]);
	  };

	  	printf("\n");*/

}



//__global__ (int * cases_number, int n)
//{

//}

void SPH_AdvanceCUDA ( float dt, float ss,float m_time )
{
//	advanceParticles<<< fcuda.numBlocks, fcuda.numThreads>>> ( bufPnts, fcuda.pnts, dt,m_time, ss );
	advanceParticles<<< fcuda.numBlocks, fcuda.numThreads>>> ( bufPnts, fcuda.pnts, dt, ss,m_time );
    CUT_CHECK_ERROR( "Kernel execution failed");
    cudaThreadSynchronize ();
}

}	