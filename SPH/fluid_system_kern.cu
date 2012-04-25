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


#define	COLLI_DET  0.0001f
#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_


#include <stdio.h>
#include "cutil_math.h"
#define EPS 0.00001

#include "fluid_system_host.cuh"

//#include <thrust/scan.h>
#define TOTAL_THREADS		65536
#define BLOCK_THREADS		256
#define MAX_NBR				80

__constant__	FluidParams		simData;		// simulation data (on device)

__device__ int				bufNeighbor[ TOTAL_THREADS*MAX_NBR ];
__device__ float			bufNdist[ TOTAL_THREADS*MAX_NBR ];	

#define COLOR(r,g,b)	( (uint((r)*255.0f)<<24) | (uint((g)*255.0f)<<16) | (uint((b)*255.0f)<<8) )
#define COLORA(r,g,b,a)	( (uint((r)*255.0f)<<24) | (uint((g)*255.0f)<<16) | (uint((b)*255.0f)<<8) | uint((a)*255.0f) )

#define NULL_HASH		333333

#define OFFSET_CLR		12
#define OFFSET_NEXT		16
#define OFFSET_VEL		20
#define OFFSET_VEVAL	32
#define OFFSET_PRESS	48
#define OFFSET_DENS		52
#define OFFSET_FORCE	56



#define	kR 4.0
#define kS 1400.0

__device__ void mul_Matrix_Vector(float * matrix ,float3 * vector, float3 * result)
{

	result->x=matrix[0]*vector->x+matrix[1]*vector->y+matrix[2]*vector->z;
	result->y=matrix[3]*vector->x+matrix[4]*vector->y+matrix[5]*vector->z;
	result->z=matrix[6]*vector->x+matrix[7]*vector->y+matrix[8]*vector->z;
}

__device__ void compute_matrix_value(float * matrix, float * result)
{
	(*result)=matrix[0]*matrix[4]*matrix[8]+
		matrix[1]*matrix[5]*matrix[6]+
		matrix[2]*matrix[3]*matrix[7]-
		matrix[2]*matrix[4]*matrix[6]-
		matrix[0]*matrix[5]*matrix[7]-
		matrix[1]*matrix[3]*matrix[8];
}




__global__ void computeDensity(float * out_density, float *in_G[], float3 * in_mean_p,char * bufPnts ,int numPnt,int x, int y,int z,float size_voxel)//
{




	uint n = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;


	float3 min_=make_float3(-25,-20,0);
	float3 max_=make_float3(25,20,40);

	uint ndz =n/(x*y);
	uint ndy = (n-ndz*x*y)/x;
	uint ndx = (n-ndz*x*y-ndy*x);




	if(ndx<x&&ndy<y&&ndz<z) 
	{

		int index=n;
		float den=0.0;
		float3 position=in_mean_p[n];

		position.x=ndx*size_voxel+min_.x;
		position.y=ndy*size_voxel+min_.y;
		position.z=ndz*size_voxel+min_.z;

		for(int i=0;i<numPnt;i++)
		{
			char * data=bufPnts + i * simData.stride;

			float3 posi = *(float3 *)data;	
			//float3 dis=(posi-position)*simData.sim_scale ;
           float3 dis=(posi-position)*simData.sim_scale;

			//float3 Gr;
			//mul_Matrix_Vector(in_G[i], & dis, &Gr);
			//float l=length(Gr);
			
			float l=length(dis);
			if(l<simData.smooth_rad)
			{
				//float G;
				//compute_matrix_value(in_G[i],& G);

			//	float m=simData.smooth_rad-length(Gr);
				float m=(simData.smooth_rad-l)/simData.smooth_rad;

				den+=m*m*m*m/(*(float*) (data + OFFSET_DENS));
			}

		}
		
		den*=simData.pmass*simData.surface_kernel;

		out_density[index]=den;
		/*	if(position.z>10)
		out_density[index]=1;
		else
		out_density[index]=0.0;*/


	}


}






__global__ void compute_vertices_number(int * case_number,int * vertices_number, float * color_density, int * d_numVertsTable,  int x, int y, int z, float isolevel)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if(index<x*y*z)
	{
		vertices_number[index]=0;
	}
	else 
		return;

	uint ndz =index/(x*y);
	uint ndy = (index-ndz*x*y)/x;
	uint ndx = (index-ndz*x*y-ndy*x);


	if(ndx<x-1&&ndx>0&&ndy<y-1&&ndy>0&&ndz<z-1&&ndz>0)
	{

		//no intersections
		float color[8];
		color[0]=color_density[index];
		color[1]=color_density[index+x];
		color[2]=color_density[index+x+1];
		color[3]=color_density[index+1];
		color[4]=color_density[index+x*y];
		color[5]=color_density[index+x+x*y];
		color[6]=color_density[index+x+1+x*y];
		color[7]=color_density[index+1+x*y];


		uint cubeindex=0;
		if(color[0]<isolevel) cubeindex+=1;
		if(color[1]<isolevel) cubeindex+=2;
		if(color[2]<isolevel) cubeindex+=4;
		if(color[3]<isolevel) cubeindex+=8;
		if(color[4]<isolevel) cubeindex+=16;
		if(color[5]<isolevel) cubeindex+=32;
		if(color[6]<isolevel) cubeindex+=64;
		if(color[7]<isolevel) cubeindex+=128;

		case_number[index]=cubeindex;
		vertices_number[index]=d_numVertsTable[cubeindex]; 





	}

}


__device__ void computeVetexNormal(int edge, float * color_density, int ** d_edgeVertices, float3 * v,float3 * normals, float3 * normal, float3 * position, float size_voxel, float  isolevels)
{

	int x=d_edgeVertices[edge][0];
	int y=d_edgeVertices[edge][1];
	float density1=color_density[x];
	float density2=color_density[y];

	//float3 normal1=normals[x];
	//float3 normal2=normals[y];

	float3 v1=v[x];
	float3 v2=v[y];

	if(fabs(density2-isolevels)<EPS)
	{
	//	*normal=normal2;
		*position=v2;
		return;
	}
	if(abs(density1-isolevels)<EPS)
	{
	//	*normal=normal1;
		*position=v1;
		return;
	}

	float weight=(isolevels-density2)/(density1-density2);

	//*normal=lerp(normal1,normal2,weight);
	*position=lerp(v2,v1,weight);


}



__global__ void marching_cube(float3 * points, float3 * normals, int * d_numVertsTable, int ** d_edgeVertices, int ** d_triTable, float * color_density,int x, int y, int z, float size_voxel, float isolevels, int * vertices_number,int * case_number )
{

	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	int ndz =index/(x*y);
	int ndy = (index-ndz*x*y)/x;
	int ndx = (index-ndz*x*y-ndy*x);



	float3 min_=make_float3(-25,-20,0);
	//float3 max_=make_float3(25,20,40);

	if(ndx<x-1&&ndx>0&&ndy<y-1&&ndy>0&&ndz<z-1&&ndz>0)
	{


		int cases=case_number[index];
		int numberVer=d_numVertsTable[cases];

		if(numberVer==0)
			return;

		float color[8];
		color[0]=color_density[index];
		color[1]=color_density[index+x];
		color[2]=color_density[index+x+1];
		color[3]=color_density[index+1];
		color[4]=color_density[index+x*y];
		color[5]=color_density[index+x+x*y];
		color[6]=color_density[index+x+1+x*y];
		color[7]=color_density[index+1+x*y];



		//compute vertices positon and normals

		float3 v[8];

		v[0].x=ndx*size_voxel+min_.x;
		v[0].y=ndy*size_voxel+min_.y;
		v[0].z= ndz*size_voxel+min_.z;

		v[1].x=ndx*size_voxel+min_.x;
		v[1].y=(ndy+1)*size_voxel+min_.y;
		v[1].z=ndz*size_voxel+min_.z;



		v[2].x=(ndx+1)*size_voxel+min_.x;
		v[2].y=(ndy+1)*size_voxel+min_.y;
		v[2].z=ndz*size_voxel+min_.z;


		v[3].x=(ndx+1)*size_voxel+min_.x;
		v[3].y=ndy*size_voxel+min_.y;
		v[3].z=ndz*size_voxel+min_.z;

		v[4].x=ndx*size_voxel+min_.x;
		v[4].y=ndy*size_voxel+min_.y;
		v[4].z= (ndz+1)*size_voxel+min_.z;

		v[5].x=ndx*size_voxel+min_.x;
		v[5].y=(ndy+1)*size_voxel+min_.y;
		v[5].z=(ndz+1)*size_voxel+min_.z;



		v[6].x=(ndx+1)*size_voxel+min_.x;
		v[6].y=(ndy+1)*size_voxel+min_.y;
		v[6].z=(ndz+1)*size_voxel+min_.z;


		v[7].x=(ndx+1)*size_voxel+min_.x;
		v[7].y=ndy*size_voxel+min_.y;
		v[7].z=(ndz+1)*size_voxel+min_.z;






		float3 normal[8];
		/*///add two columns and rows to the 


		normal[0].x=(color_density[index+1]-color_density[index-1])/size_voxel;
		normal[0].y=(color_density[index+x]-color_density[index-x])/size_voxel;
		normal[0].z=(color_density[index+x*y]-color_density[index-x*y])/size_voxel;


		normal[1].x=(color_density[index+1+x]-color_density[index-1+x])/size_voxel;
		normal[1].y=(color_density[index+x+x]-color_density[index-x+x])/size_voxel;
		normal[1].z=(color_density[index+x*y+x]-color_density[index-x*y+x])/size_voxel;                 



		normal[2].x=(color_density[index+1+x+1]-color_density[index-1+x+1])/size_voxel;
		normal[2].y=(color_density[index+x+x+1]-color_density[index-x+x+1])/size_voxel;
		normal[2].z=(color_density[index+x*y+x+1]-color_density[index-x*y+x+1])/size_voxel; 




		normal[3].x=(color_density[index+1+1]-color_density[index-1+1])/size_voxel;
		normal[3].y=(color_density[index+x+1]-color_density[index-x+1])/size_voxel;
		normal[3].z=(color_density[index+x*y+1]-color_density[index-x*y+1])/size_voxel; 



		normal[4].x=(color_density[index+1+x*y]-color_density[index-1+x*y])/size_voxel;
		normal[4].y=(color_density[index+x+x*y]-color_density[index-x+x*y])/size_voxel;
		normal[4].z= (color_density[index+x*y+x*y]-color_density[index-x*y+x*y])/size_voxel; 



		normal[5].x=(color_density[index+1+x+x*y]-color_density[index-1+x+x*y])/size_voxel;
		normal[5].y=(color_density[index+x+x+x*y]-color_density[index-x+x+x*y])/size_voxel;
		normal[5].z=(color_density[index+x*y+x+x*y]-color_density[index-x*y+x+x*y])/size_voxel; 



		normal[6].x=(color_density[index+1+x+1+x*y]-color_density[index-1+x+1+x*y])/size_voxel;
		normal[6].y=(color_density[index+x+x+1+x*y]-color_density[index-x+x+1+x*y])/size_voxel;
		normal[6].z=(color_density[index+x*y+x+1+x*y]-color_density[index-x*y+x+1+x*y])/size_voxel; 




		normal[7].x=(color_density[index+1+1+x*y]-color_density[index-1+1+x*y])/size_voxel;
		normal[7].y=(color_density[index+x+1+x*y]-color_density[index-x+1+x*y])/size_voxel;
		normal[7].z=(color_density[index+x*y+1+x*y]-color_density[index-x*y+1+x*y])/size_voxel; */




		//interplate 

		int ver_index=vertices_number[index];
		int i=0;


		while(numberVer!=0)
		{

			if(d_triTable[cases][i]!=-1)//////
			{

				//computeVetexNormal(d_triTable[cases][i], color, d_edgeVertices, v, normal, &normals[ver_index++], &points[ver_index++], size_voxel, isolevels);
				computeVetexNormal(d_triTable[cases][i], color, d_edgeVertices, v, normal, &normals[ver_index], &points[ver_index], size_voxel, isolevels);
				ver_index++;
				numberVer--;
			}
			i++;

		}

	}
}













__device__ void computeGi( float * B)
{
	int m=3,n=3;
	// particle index

	float  S[3];
	float  U[9];
	float  V[9];
	float e[3];
	float work[3];


int nct = min( m-1, n );
    int nrt = max( 0, n-2 );
    int i=0,
        j=0,
        k=0;

    for( k=0; k<max(nct,nrt); ++k )
    {
        if( k < nct )
        {
            // Compute the transformation for the k-th column and
            // place the k-th diagonal in s[k].
            // Compute 2-norm of k-th column without under/overflow.
            S[k] = 0;
            for( i=k; i<m; ++i )
                 S[k] = hypot( S[k], B[i*n+k] );

            if( S[k] != 0 )
            {
                if( B[k*n+k] < 0 )
                    S[k] = -S[k];

                for( i=k; i<m; ++i )
                    B[i*n+k] /= S[k];
                B[k*n+k] += 1;
            }
            S[k] = -S[k];
        }

        for( j=k+1; j<n; ++j )
        {
            if( (k < nct) && ( S[k] != 0 ) )
            {
                // apply the transformation
                float t = 0;
                for( i=k; i<m; ++i )
                    t += B[i*n+k] * B[i*n+j];

                t = -t / B[k*n+k];
                for( i=k; i<m; ++i )
                    B[i*n+j] += t*B[i*n+k];
            }
            e[j] = B[k*n+j];
        }


        // Place the transformation in U for subsequent back
        // multiplication.
        if( (k < nct) )
            for( i=k; i<m; ++i )
                U[i*n+k] = B[i*n+k];

        if( k < nrt )
        {
            // Compute the k-th row transformation and place the
            // k-th super-diagonal in e[k].
            // Compute 2-norm without under/overflow.
            e[k] = 0;
            for( i=k+1; i<n; ++i )
                e[k] = hypot( e[k], e[i] );

            if( e[k] != 0 )
            {
                if( e[k+1] < 0 )
                    e[k] = -e[k];

                for( i=k+1; i<n; ++i )
                    e[i] /= e[k];
                e[k+1] += 1;
            }
            e[k] = -e[k];

            if( (k+1 < m) && ( e[k] != 0 ) )
            {
                // apply the transformation
                for( i=k+1; i<m; ++i )
                    work[i] = 0;

                for( j=k+1; j<n; ++j )
                    for( i=k+1; i<m; ++i )
                        work[i] += e[j] * B[i*n+j];

                for( j=k+1; j<n; ++j )
                {
                    float t = -e[j]/e[k+1];
                    for( i=k+1; i<m; ++i )
                        B[i*n+j] += t * work[i];
                }
            }

            // Place the transformation in V for subsequent
            // back multiplication.
			
                for( i=k+1; i<n; ++i )
                    V[i*n+k] = e[i];

        }
    }

	
    // Set up the final bidiagonal matrix or order p.
	//cout<<B<<endl;
    int p = n;

    if( nct < n )
        S[nct] = B[nct*n+nct];
    if( m < p )
        S[p-1] = 0;

    if( nrt+1 < p )
        e[nrt] = B[nrt*n+p-1];
    e[p-1] = 0;

    // if required, generate U
 
        for( j=nct; j<n; ++j )
        {
            for( i=0; i<m; ++i )
                U[i*n+j] = 0;
            U[j*n+j] = 1;
        }

        for( k=nct-1; k>=0; --k )
            if( S[k] != 0 )
            {
                for( j=k+1; j<n; ++j )
                {
                    float t = 0;
                    for( i=k; i<m; ++i )
                        t += U[i*n+k] * U[i*n+j];
                    t = -t / U[k*n+k];

                    for( i=k; i<m; ++i )
                        U[i*n+j] += t * U[i*n+k];
                }

                for( i=k; i<m; ++i )
                    U[i*n+k] = -U[i*n+k];
                U[k*n+k] = 1 + U[k*n+k];

                for( i=0; i<k-1; ++i )
                    U[i*n+k] = 0;
            }
            else
            {
                for( i=0; i<m; ++i )
                    U[i*n+k] = 0;
                U[k*n+k] = 1;
            }
 

    // if required, generate V

        for( k=n-1; k>=0; --k )
        {
            if( (k < nrt) && ( e[k] != 0 ) )
                for( j=k+1; j<n; ++j )
                {
                    float t = 0;
                    for( i=k+1; i<n; ++i )
                        t += V[i*n+k] * V[i*n+j];
                    t = -t / V[(k+1)*n+k];

                    for( i=k+1; i<n; ++i )
                        V[i*n+j] += t * V[i*n+k];
                }

            for( i=0; i<n; ++i )
                V[i*n+k] = 0;
            V[k*n+k] = 1;
        }

    int pp = p-1;
    int iter = 0;
    double eps = pow( 2.0, -52.0 );

    while( p > 0 )
    {
        int k = 0;
        int kase = 0;

        // Here is where a test for too many iterations would go.
        // This section of the program inspects for negligible
        // elements in the s and e arrays. On completion the
        // variables kase and k are set as follows.
        // kase = 1     if s(p) and e[k-1] are negligible and k<p
        // kase = 2     if s(k) is negligible and k<p
        // kase = 3     if e[k-1] is negligible, k<p, and
        //				s(k), ..., s(p) are not negligible
        // kase = 4     if e(p-1) is negligible (convergence).
        for( k=p-2; k>=-1; --k )
        {
            if( k == -1 )
                break;

            if( abs(e[k]) <= eps*( abs(S[k])+abs(S[k+1]) ) )
            {
                e[k] = 0;
                break;
            }
        }

        if( k == p-2 )
            kase = 4;
        else
        {
            int ks;
            for( ks=p-1; ks>=k; --ks )
            {
                if( ks == k )
                    break;

                float t = ( (ks != p) ? abs(e[ks]) : 0 ) +
                         ( (ks != k+1) ? abs(e[ks-1]) : 0 );

                if( abs(S[ks]) <= eps*t )
                {
                    S[ks] = 0;
                    break;
                }
            }

            if( ks == k )
                kase = 3;
            else if( ks == p-1 )
                kase = 1;
            else
            {
                kase = 2;
                k = ks;
            }
        }
        k++;

        // Perform the task indicated by kase.
        switch( kase )
        {
            // deflate negligible s(p)
            case 1:
            {
                float f = e[p-2];
                e[p-2] = 0;

                for( j=p-2; j>=k; --j )
                {
                    float t = hypot( S[j], f );
                    float cs = S[j] / t;
                    float sn = f / t;
                    S[j] = t;

                    if( j != k )
                    {
                        f = -sn * e[j-1];
                        e[j-1] = cs * e[j-1];
                    }
                        for( i=0; i<n; ++i )
                        {
                            t = cs*V[i*n+j] + sn*V[i*n+p-1];
                            V[i*n+p-1] = -sn*V[i*n+j] + cs*V[i*n+p-1];
                            V[i*n+j] = t;
                        }
                }
            }
            break;

            // split at negligible s(k)
            case 2:
            {
                float f = e[k-1];
                e[k-1] = 0;

                for( j=k; j<p; ++j )
                {
                    float t = hypot( S[j], f );
                    float cs = S[j] / t;
                    float sn = f / t;
                    S[j] = t;
                    f = -sn * e[j];
                    e[j] = cs * e[j];

               
                        for( i=0; i<m; ++i )
                        {
                            t = cs*U[i*n+j] + sn*U[i*n+k-1];
                            U[i*n+k-1] = -sn*U[i*n+j] + cs*U[i*n+k-1];
                            U[i*n+j] = t;
                        }
                }
            }
            break;

            // perform one qr step
            case 3:
            {
                // calculate the shift
                float scale = max( max( max( max(
                             abs(S[p-1]), abs(S[p-2]) ), abs(e[p-2]) ),
                             abs(S[k]) ), abs(e[k]) );
                float sp = S[p-1] / scale;
                float spm1 = S[p-2] / scale;
                float epm1 = e[p-2] / scale;
                float sk = S[k] / scale;
                float ek = e[k] / scale;
                float b = ( (spm1+sp)*(spm1-sp) + epm1*epm1 ) / 2.0;
                float c = (sp*epm1) * (sp*epm1);
                float shift = 0;

                if( ( b != 0 ) || ( c != 0 ) )
                {
                    shift = sqrt( b*b+c );
                    if( b < 0 )
                        shift = -shift;
                    shift = c / ( b+shift );
                }
                float f = (sk+sp)*(sk-sp) + shift;
                float g = sk * ek;

                // chase zeros
                for( j=k; j<p-1; ++j )
                {
                    float t = hypot( f, g );
                    float cs = f / t;
                    float sn = g / t;
                    if( j != k )
                        e[j-1] = t;

                    f = cs*S[j] + sn*e[j];
                    e[j] = cs*e[j] - sn*S[j];
                    g = sn * S[j+1];
                    S[j+1] = cs * S[j+1];

                    //Forward transformation of YT
			
                        for( i=0; i<n; ++i )
                        {
                            t = cs*V[i*n+j] + sn*V[i*n+j+1];
                            V[i*n+j+1] = -sn*V[i*n+j] + cs*V[i*n+j+1];
                            V[i*n+j] = t;
                        }

                    t = hypot( f, g );
                    cs = f / t;
                    sn = g / t;
                    S[j] = t;
                    f = cs*e[j] + sn*S[j+1];
                    S[j+1] = -sn*e[j] + cs*S[j+1];
                    g = sn * e[j+1];
                    e[j+1] = cs * e[j+1];

                    if( ( j < m-1 ) )
                        for( i=0; i<m; ++i )
                        {
                            t = cs*U[i*n+j] + sn*U[i*n+j+1];
                            U[i*n+j+1] = -sn*U[i*n+j] + cs*U[i*n+j+1];
                            U[i*n+j] = t;
                        }
                }
                e[p-2] = f;
                iter = iter + 1;
            }
            break;

            // convergence
            case 4:
            {
                // Make the singular values positive.
                if( S[k] <= 0 )
                {
                    S[k] = ( S[k] < 0 ) ? -S[k] : 0;
                 
                        for( i=0; i<=pp; ++i )
                            V[i*n+k] = -V[i*n+k];
                }

                // Order the singular values.
                while( k < pp )
                {
                    if( S[k] >= S[k+1] )
                        break;

                    float t = S[k];
                    S[k] = S[k+1];
                    S[k+1] = t;

                    if(( k < n-1 ) )
                        for( i=0; i<n; ++i )
                        {
                           float temp=V[i*n+k];
                            V[i*n+k]=V[i*n+k+1];
                            V[i*n+k+1] =temp;
                     
                         };

                    if( ( k < m-1 ) )
                        for( i=0; i<m; ++i )
                        {
                        
                            float temp=U[i*n+k];
                            U[i*n+k]=U[i*n+k+1];
                            U[i*n+k+1] =temp;
                            
                         }
                    k++;
                }
                iter = 0;
                p--;
            }
            break;
        }
    }
	

		//float kn=.5;

		for(int i=2;i<3;i++)
		{
			S[i]=1/(max(S[i],S[1]/kR));
		}

		for(int i=0;i<9;i++)
		{
			B[i]=0;
		}
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				for(int k=0;k<3;k++)
				{
					B[i*3+j]+=S[k]*U[i*3+k]*V[k*3+j];
				}
			}
		}



}












__global__ void computePressure ( char* bufPnts, int numPnt) // bufPnts = mBuf[0].data;
{
	uint ndx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	// particle index

	if(ndx<numPnt)
	{
		float3* pos = (float3*) (bufPnts + __mul24(ndx, simData.stride));	
		float3* posi;

		char *dat2;
		//char *length;
		//Fluid *p, *q;
		int cnt = 0;
		float3 dist;
		double  sum, dsq, c;
		double d,/* d2,*/ mR, mR2;
		d = simData.sim_scale;
		mR =simData.smooth_rad;
		mR2 = mR*mR;	

		sum = 0.0;
		for(int i = 0; i < numPnt; i ++){
			dat2 = bufPnts + i*simData.stride;
			posi = (float3*)(dat2);


			if ( pos==posi ) continue;
			//	dist.x = ( pos->x - posi->x)*d;		// dist in cm
			//	dist.y = ( pos->y - posi->y)*d;
			//	dist.z = ( pos->z - posi->z)*d;
			//dist = (pos - posi)*make_float3(d);
			dist = (*pos - *posi)*d;
			dsq = dot(dist,dist);			
			if ( mR2 > dsq ) {
				c =  mR2 - dsq;
				sum += c * c * c;
				cnt++;
				//if ( p == m_CurrP ) q->tag = true;
			}
		}	
		sum = sum * simData.pmass * simData.poly6kern;
		if( sum ==0.0)
			sum = 1.0;
		*(float*) ((char*)pos + OFFSET_PRESS) = ( sum - simData.rest_dens ) * simData.stiffness;
		*(float*) ((char*)pos + OFFSET_DENS) = 1.0f / sum;	


	}


	//}		
	//__syncthreads ();
}


__global__ void computeForce ( char* bufPnts,int numPnt, float ** out_matrix, float3 * mean_points)
{
	uint ndx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index		
	if ( ndx < numPnt ) {

		float3* pos = (float3*) (bufPnts + __mul24(ndx, simData.stride));	
		float3* posi;
		float3 force;



		register double termPressure, termVelo, termDist;
		double /*c,*/ r, /*d, sum,*/ dsq;

		float3 dist;
		double mR, mR2;
		float press = *(float*) ((char*)pos + OFFSET_PRESS);
		float dens = *(float*) ((char*)pos + OFFSET_DENS);
		float3 veval = *(float3*) ((char*)pos + OFFSET_VEVAL );
		float3 qeval;


		mR =simData.smooth_rad;// m_Param[SPH_SMOOTHRADIUS];
		mR2 = (mR*mR);

		termVelo = simData.lapkern * simData.visc;


		force = make_float3(0,0,0);

		float weight_sum = 0.0;
		float3 mean_point=make_float3(0,0,0);

		//float weights[2047];


		int near_by=0;
		for(int i = 0; i<numPnt;i++)
		{
			posi = (float3 *)(bufPnts + i * simData.stride);

			//weights[i]=0.0;

			if ( pos == posi )
			{
			//	weights[i]=1.0;
				weight_sum+=1.0;
				mean_point+=(*posi);
				continue;
			}

			dist = (*pos - *posi)*simData.sim_scale;

			dsq = dot(dist,dist);//(dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( mR2 > dsq ) {
				r = sqrtf ( dsq );
				near_by++;
				//Force
				termPressure = -0.5f * (mR - r) * simData.spikykern * ( press + *(float*)((char*)posi+OFFSET_PRESS)) / r;
				termDist = (mR - r) * dens * *(float*)((char*)posi+OFFSET_DENS);	
				qeval = *(float3*)((char*)posi+OFFSET_VEVAL);
				force.x += ( termPressure * dist.x + termVelo * (qeval.x - veval.x) ) * termDist;
				force.y += ( termPressure * dist.y + termVelo * (qeval.y - veval.y) ) * termDist;
				force.z += ( termPressure * dist.z + termVelo * (qeval.z - veval.z) ) * termDist;

				//MeanPoint
				float weight=1.0-r*r*r/mR/mR/mR;
				mean_point+=weight*(*posi);
				weight_sum+=weight;

			//	weights[i]=weight;


			}
		}


		*(float3*) ((char*)pos + OFFSET_FORCE ) = force;	

		///// mean matri
		if (weight_sum<pow( 2.0, -10))
			mean_point=*pos;
		else
			mean_point/=weight_sum;
		mean_points[ndx]=mean_point;



		//float *out=out_matrix[ndx];
	/*	float out[9];
		for(int i=0;i<9;i++)
		{
			out[i]=0.0;
		}

		for(int i=0;i<numPnt;i++)
		{  

			float3 data = *((float3 *)(bufPnts+__mul24(i, simData.stride)));
			data=data-mean_point;
			out[0]+=weights[i]*data.x*data.x;
			out[1]+=weights[i]*data.x*data.y;
			out[2]+=weights[i]*data.x*data.z;
			out[3]+=weights[i]*data.y*data.x;
			out[4]+=weights[i]*data.y*data.y;
			out[5]+=weights[i]*data.y*data.z;
			out[6]+=weights[i]*data.z*data.x;
			out[7]+=weights[i]*data.z*data.y;
			out[8]+=weights[i]*data.z*data.z;

			out[0]=1;
			out[1]=2;
			out[2]=3;
			out[3]=4;
			out[4]=5;
			out[5]=6;
			out[6]=7;
			out[7]=8;
			out[8]=9;

		}


	//	if(near_by>25)
		{
			/*if(weight_sum>pow( 2.0, -5))
			{
				for(int i=0;i<9;i++)
				{
					out[i]/=weight_sum;
					//out[i]=weights[i];
				}
			}*/


	/*		computeGi(out);
		

			float * o=out_matrix[ndx];
			for(int i=0;i<9;i++)
			{
				o[i]=out[i];

			}

		}
	/*	else
		{
			int Ks=1400;
			for(int i=0;i<3;i++)
			{
				for(int j=0;j<3;j++)
				{
					if(i==j)
						out[i*3+i]=Ks;
					else
						out[i*3+j]=0;
				}

			}

			float * o=out_matrix[ndx];

			for(int i=0;i<9;i++)
			{
				o[i]=out[i];

			}


		}*/



	}

}


	__global__ void advanceParticles ( char* bufPnts, int numPnt, float dt,float ss,float time )
	{		

		//	uint ndx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index		

		/*		if ( ndx < numPnt ) {
		char *ptnData_start, *ptnData_end;
		//			Fluid* pnt;
		float3 norm, z, dir, acc, vnext, min, max;

		double adj;
		float speedLimit, speedLimit2, simScale, particleRadius;
		float stiff, damp, speed, diff; 
		speedLimit = simData.speedLimit;
		speedLimit2 = speedLimit*speedLimit;

		stiff = simData.stiffness;
		damp = simData.damp;
		particleRadius = simData.particleRadius;// m_Param[SPH_PRADIUS];
		min = simData.min;//m_Vec[SPH_VOLMIN];
		max = simData.max;//m_Vec[SPH_VOLMAX];
		simScale = ss;//m_Param[SPH_SIMSCALE];

		float3* pos = (float3*) (bufPnts + __mul24(ndx, simData.stride));	
		float3* veval =(float3*) ((char*)pos + OFFSET_VEVAL );
		float3* vel = (float3*) ((char*)pos + OFFSET_VEL );


		// Compute acceration		
		acc = *(float3*) ((char*)pos + OFFSET_FORCE );

		//acc *= simData.pmass;
		//	acc *= (1/m_Param[SPH_PMASS]);

		// Velocity limiting 
		speed = acc.x*acc.x + acc.y*acc.y + acc.z*acc.z;
		if ( speed > speedLimit2 ) {
		acc.x *= speedLimit / sqrt(speed);
		acc.y *= speedLimit / sqrt(speed);
		acc.z *= speedLimit / sqrt(speed);


		}



		// Boundary Conditions

		// Z-axis walls
		diff = 2 * particleRadius - ( pos->z - min.z - (pos->x - min.x) * simData.slope )*simScale;
		if (diff > COLLI_DET ) {			

		norm = make_float3 ( -simData.slope, 0, 1.0 - simData.slope );
		adj = stiff * diff - damp * (norm.x*veval.x+norm.y*veval.y+norm.z+veval.z);//norm.Dot ( veval );
		acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}		

		diff = 2 * particleRadius - ( max.z - pos->z )*simScale;
		if (diff > COLLI_DET) {
		norm = make_float3 ( 0, 0, -1 );
		adj = stiff * diff - damp * (norm.x*veval.x+norm.y*veval.y+norm.z*veval.z);//norm.Dot ( pnt->vel_eval );
		acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}

		// X-axis walls

		diff = 2 * particleRadius - ( pos->x - min.x + (sin(time*10.0)-1+(pos->y*0.025)*0.25) * simData.xminSin )*simScale;	
		//diff = 2 * particleRadius - ( pnt->pos.x - min.x + (sin(time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*simScale;	
		if (diff > COLLI_DET ) {
		norm = make_float3( 1.0, 0, 0 );
		adj = (simData.xminSin + 1) * stiff * diff - damp * (norm.x*veval.x+norm.y*veval.y+norm.z*veval.z);//norm.Dot ( pnt->vel_eval ) ;
		acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;					
		}

		diff = 2 * particleRadius - ( max.x - pos->x + (sin(time*10.0)-1) * simData.xmaxSin )*simScale;	
		if (diff > COLLI_DET) {
		norm = make_float3( -1, 0, 0 );
		adj = (simData.xmaxSin+1) * stiff * diff - damp * (norm.x*veval.x+norm.y*veval.y+norm.z*veval.z);//.Dot ( pnt->vel_eval );
		acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}


		// Y-axis walls
		diff = 2 * particleRadius - ( pos->y - min.y )*simScale;			
		if (diff > COLLI_DET) {
		norm = make_float3 ( 0, 1, 0 );
		adj = stiff * diff - damp * (norm.x*veval.x+norm.y*veval.y+norm.z*veval.z);//norm.Dot ( pnt->vel_eval );
		acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}
		diff = 2 * particleRadius - ( max.y - pos->y )*simScale;
		if (diff > COLLI_DET) {
		norm = make_float3 ( 0, -1, 0 );
		adj = stiff * diff - damp * (norm.x*veval.x+norm.y*veval.y+norm.z*veval.z);//norm.Dot ( pnt->vel_eval );
		acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}

		// Plane gravity
		//if ( simData.gravity > 0) 

		acc.z += simData.gravity;



		// Leapfrog Integration ----------------------------
		vnext = acc;							
		vnext.x *= dt;
		vnext.y *= dt;
		vnext.z *= dt;
		vnext.x += vel.x;						// v(t+1/2) = v(t-1/2) + a(t) dt
		vnext.y += vel.y;
		vnext.z += vel.z;
		veval = vel;
		veval += vnext;
		veval *= 0.5;					// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
		vel = vnext;
		vnext *= dt/simScale;
		pos += vnext;						// p(t+1) = p(t) + v(t+1/2) dt
		*/
		/*		if ( m_Param[CLR_MODE]==1.0 ) {
		adj = fabs(vnext.x)+fabs(vnext.y)+fabs(vnext.z) / 7000.0;
		adj = (adj > 1.0) ? 1.0 : adj;
		pnt->clr = COLORA( 0, adj, adj, 1 );
		}
		if ( m_Param[CLR_MODE]==2.0 ) {
		float v = 0.5 + ( pnt->pressure / 1500.0); 
		if ( v < 0.1 ) v = 0.1;
		if ( v > 1.0 ) v = 1.0;
		pnt->clr = COLORA ( v, 1-v, 0, 1 );
		}
		*/

		// Euler integration -------------------------------
		/* acc += m_Gravity;
		acc *= m_DT;
		pnt->vel += acc;				// v(t+1) = v(t) + a(t) dt
		pnt->vel_eval += acc;
		pnt->vel_eval *= m_DT/d;
		pnt->pos += pnt->vel_eval;
		pnt->vel_eval = pnt->vel;  */	


		/*				if ( m_Toggle[WRAP_X] ) {
		diff = pnt->pos.x - (m_Vec[SPH_VOLMIN].x + 2);			// -- Simulates object in center of flow
		if ( diff <= 0 ) {
		pnt->pos.x = (m_Vec[SPH_VOLMAX].x - 2) + diff*2;				
		pnt->pos.z = 10;
		}
		}	
		}
		*/
		//	time += m_DT;




		uint ndx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index		

		if ( ndx < numPnt ) {

			// Get particle vars
			float3* pos = (float3*) (bufPnts + __mul24(ndx, simData.stride));			
			float3* vel = (float3*) ((char*)pos + OFFSET_VEL );
			float3* veval = (float3*) ((char*)pos + OFFSET_VEVAL );
			float3 accel = *(float3*) ((char*)pos + OFFSET_FORCE );
			float3 vcurr, vnext;	
			float3 norm,/* z, dir,*/ min, max;

			double adj;
			float speedLimit, speedLimit2, simScale, particleRadius;
			float stiff, damp, speed, diff; 
			speedLimit = 200.0;//simData.speedLimit;
			speedLimit2 = speedLimit*speedLimit;
			particleRadius = simData.particleRadius;
			stiff = 10000.0;//simData.stiffness;
			damp =  256;//simData.damp;
			simScale = simData.sim_scale;
			min = /*simData.min;*/make_float3(-25,-20,0);
			max = /*simData.max;*/make_float3(25,20,40);//simData.max;


			// Leapfrog integration						
			accel.x *= 0.00020543;			// NOTE - To do: SPH_PMASS should be passed in			
			accel.y *= 0.00020543;
			accel.z *= 0.00020543;	
			speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
			if ( speed > speedLimit2 ) {
				accel.x *= speedLimit/sqrt(speed);
				accel.y *= speedLimit/sqrt(speed);
				accel.z *= speedLimit/sqrt(speed);


			}
			diff = 2 * particleRadius - ( pos->z - min.z - (pos->x - min.x) *simData.slope )*simScale;
			if (diff > COLLI_DET ) {			

				norm = make_float3 ( -simData.slope, 0, 1.0 - simData.slope );
				adj = stiff * diff - damp * (norm.x*veval->x+norm.y*veval->y+norm.z*veval->z);//norm.Dot ( veval );
				//accel.x = -accel.x;accel.y=-accel.y;accel.z=-accel.z; 
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}		

			diff = 2 * particleRadius - ( max.z - pos->z )*simScale;
			if (diff > COLLI_DET) {
				norm = make_float3 ( 0, 0, -1 );
				adj = stiff * diff - damp * (norm.x*veval->x+norm.y*veval->y+norm.z*veval->z);//norm.Dot ( pnt->vel_eval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}

			// X-axis walls

			diff = 2 * particleRadius - ( pos->x - min.x + (sin(time*10.0)-1+(pos->y*0.025)*0.25) * simData.xminSin )*simScale;	
			//diff = 2 * particleRadius - ( pnt->pos.x - min.x + (sin(time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*simScale;	
			if (diff > COLLI_DET ) {
				norm = make_float3( 1.0, 0, 0 );
				adj = (simData.xminSin + 1) * stiff * diff - damp * (norm.x*veval->x+norm.y*veval->y+norm.z*veval->z);//norm.Dot ( pnt->vel_eval ) ;
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}

			diff = 2 * particleRadius - ( max.x - pos->x + (sin(time*10.0)-1) * simData.xmaxSin )*simScale;	
			if (diff > COLLI_DET) {
				norm = make_float3( -1, 0, 0 );
				adj = (simData.xmaxSin+1) * stiff * diff - damp * (norm.x*veval->x+norm.y*veval->y+norm.z*veval->z);//.Dot ( pnt->vel_eval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}


			// Y-axis walls
			diff = 2 * particleRadius - ( pos->y - min.y )*simScale;			
			if (diff > COLLI_DET) {
				norm = make_float3 ( 0, 1, 0 );
				adj = stiff * diff - damp * (norm.x*veval->x+norm.y*veval->y+norm.z*veval->z);//norm.Dot ( pnt->vel_eval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
			diff = 2 * particleRadius - ( max.y - pos->y )*simScale;
			if (diff > COLLI_DET) {
				norm = make_float3 ( 0, -1, 0 );
				adj = stiff * diff - damp * (norm.x*veval->x+norm.y*veval->y+norm.z*veval->z);//norm.Dot ( pnt->vel_eval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}		

			accel.z -= 9.8;	
			vcurr = *vel;
			vnext.x = accel.x*dt+vcurr.x;
			vnext.y = accel.y*dt+vcurr.y;
			vnext.z = accel.z*dt+vcurr.z;

			accel.x = (vcurr.x+vnext.x)*0.5;			
			accel.y = (vcurr.y+vnext.y)*0.5;			
			accel.z = (vcurr.z+vnext.z)*0.5;
			*veval = accel;
			*vel =vnext;
			dt /= simData.sim_scale;
			vnext.x = pos->x + vnext.x*dt;
			vnext.y = pos->y + vnext.y*dt;
			vnext.z = pos->z + vnext.z*dt;
			*pos = vnext;

		}

		/*
		vcurr = *vel;
		vnext.x = accel.x*dt + vcurr.x;	
		vnext.y = accel.y*dt + vcurr.y;	
		vnext.z = accel.z*dt + vcurr.z;			// v(t+1/2) = v(t-1/2) + a(t) dt			

		accel.x = (vcurr.x + vnext.x) * 0.5;		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
		accel.y = (vcurr.y + vnext.y) * 0.5;		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
		accel.z = (vcurr.z + vnext.z) * 0.5;		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later

		*veval = accel;			
		*vel = vnext;

		dt /= simData.sim_scale;
		vnext.x = pos->x + vnext.x*dt;
		vnext.y = pos->y + vnext.y*dt;
		vnext.z = pos->z + vnext.z*dt;
		*pos = vnext;						// p(t+1) = p(t) + v(t+1/2) dt			
		}	
		*/
		__syncthreads ();
	}

#endif
