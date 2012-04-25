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
//	#include <math.h>
	#include "cutil_math.h"

	#include "fluid_system_host.cuh"

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
	


	

	

		
	__global__ void computePressure ( char* bufPnts, int numPnt ) // bufPnts = mBuf[0].data;
	{
		uint ndx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
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
	//	d2 = d*d;
		mR =simData.smooth_rad;
		mR2 = mR*mR;	
//		dat = bufPnts +ndx;

		//length = bufPnts + numPnt*simData.stride;
		sum = 0.0;
		//for ( dat2 = bufPnts; dat2 < length; dat2 += simData.stride ) {
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
		
			
			
			
		
		//}		
		//__syncthreads ();
	}


	
	__global__ void computeForce ( char* bufPnts,int numPnt )
	{
		uint ndx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index		
		//if ( ndx < numPnt ) {
		
		float3* pos = (float3*) (bufPnts + __mul24(ndx, simData.stride));	
		float3* posi;
		float3 force;

		
		char *dat2;//, *dat2_end;
		//Fluid *p, *q;
	//	Vector3DF force, fcurr;
	//	register double termPressure, termVelo, termDist;
		register double termPressure, termVelo, termDist;
		double /*c,*/ r, /*d, sum,*/ dsq;
		//double dx, dy, dz;
		float3 dist;
		double mR, mR2;
		float press = *(float*) ((char*)pos + OFFSET_PRESS);
		float dens = *(float*) ((char*)pos + OFFSET_DENS);
		float3 veval = *(float3*) ((char*)pos + OFFSET_VEVAL );
		float3 qeval;
		
	//	d = simData.sim_scale;//m_Param[SPH_SIMSCALE];
		mR =simData.smooth_rad;// m_Param[SPH_SMOOTHRADIUS];
		mR2 = (mR*mR);
	//	visc = simData.visc;//m_Param[SPH_VISC];
		termVelo = simData.lapkern * simData.visc;
		


		//dat1_end = bufPnts + numPnt*simData.stride;
		
	//	sum = 0.0;
		force = make_float3(0,0,0);
		
	//	dat2_end = bufPnts + numPnt*simData.stride;
	//	for ( dat2 = bufPnts; dat2 < dat2_end; dat2 += simData.stride ) {
		for(int i = 0; i<numPnt;i++){
			dat2 = bufPnts + i * simData.stride;
			posi = (float3*)dat2;
			
			

			if ( pos == posi ) continue;
		//	dist.x = ( pos->x - posi->x )*simData.sim_scale;			
		//	dist.y = ( pos->y - posi->y )*simData.sim_scale;
		//	dist.z = ( pos->z - posi->z )*simData.sim_scale;
			dist = (*pos - *posi)*simData.sim_scale;

			dsq = dot(dist,dist);//(dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( mR2 > dsq ) {
				r = sqrtf ( dsq );
			//	c = (mR - r);
				termPressure = -0.5f * (mR - r) * simData.spikykern * ( press + *(float*)((char*)posi+OFFSET_PRESS)) / r;
				termDist = (mR - r) * dens * *(float*)((char*)posi+OFFSET_DENS);	
				qeval = *(float3*)((char*)posi+OFFSET_VEVAL);
				force.x += ( termPressure * dist.x + termVelo * (qeval.x - veval.x) ) * termDist;
				force.y += ( termPressure * dist.y + termVelo * (qeval.y - veval.y) ) * termDist;
				force.z += ( termPressure * dist.z + termVelo * (qeval.z - veval.z) ) * termDist;
			}
		}			
		*(float3*) ((char*)pos + OFFSET_FORCE ) = force;	
	
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
			speedLimit = simData.speedLimit;
			speedLimit2 = speedLimit*speedLimit;
			particleRadius = simData.particleRadius;
			stiff = 10000;//simData.stiffness;
			damp =  simData.damp;
			simScale = simData.sim_scale;
			min = simData.min;//*/make_float3(-25,-25,0);
			max = simData.max;//*/make_float3(25,25,40);//simData.max;


		
			accel.x *= simData.pmass;//0.00020543;			
			accel.y *= simData.pmass;//0.00020543;
			accel.z *= simData.pmass;//0.00020543;	
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
			
			accel.z += simData.gravity;	
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
