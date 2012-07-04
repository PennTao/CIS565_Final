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
#include <conio.h>

#include <gl/glut.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/functional.h>
#include "common_defs.h"
#include "mtime.h"
#include "SPH_sim.h"
#include "scann.cuh"
#include "fluid_system_host.cuh"
//#include <thrust/scan.h>
#define COLLI_DETECT	0.00001f


extern int max_triangle_number;
extern float3 * pointsList;
extern float3 * normalList;

		int triangle_number=0;
		float3 * points_List=new float3[max_triangle_number];

FluidSystem::FluidSystem ()
{
		
		
}

void FluidSystem::Initialize ( int mode, int total )
{
	if ( mode != BFLUID ) {
		printf ( "ERROR: FluidSystem not initialized as BFLUID.\n");
	}
	PointSet::Initialize ( mode, total );

	FreeBuffers ();
	AddBuffer ( BFLUID, sizeof ( Fluid ), total );
	AddAttribute ( 0, "pos", sizeof ( Vector3DF ), false );	
	AddAttribute ( 0, "color", sizeof ( DWORD ), false );
	AddAttribute ( 0, "vel", sizeof ( Vector3DF ), false );
	AddAttribute ( 0, "ndx", sizeof ( unsigned short ), false );
	AddAttribute ( 0, "age", sizeof ( unsigned short ), false );

	AddAttribute ( 0, "pressure", sizeof ( double ), false );
	AddAttribute ( 0, "density", sizeof ( double ), false );
	AddAttribute ( 0, "sph_force", sizeof ( Vector3DF ), false );
	AddAttribute ( 0, "next", sizeof ( Fluid* ), false );
	AddAttribute ( 0, "tag", sizeof ( bool ), false );		

	SPH_Setup ();
	Reset ( total );	
}

void FluidSystem::Reset ( int nmax )
{
	ResetBuffer ( 0, nmax );

	m_DT = 0.003; //  0.001;			// .001 = for point grav

	// Reset parameters
	m_Param [ MAX_FRAC ] = 1.0;
	m_Param [ POINT_GRAV ] = 0.0;
	m_Param [ PLANE_GRAV ] = 1.0;

	m_Param [ BOUND_ZMIN_SLOPE ] = 0.0;
	m_Param [ FORCE_XMAX_SIN ] = 0.0;
	m_Param [ FORCE_XMIN_SIN ] = 0.0;	
	m_Toggle [ WRAP_X ] = false;
	m_Toggle [ WALL_BARRIER ] = false;
	m_Toggle [ LEVY_BARRIER ] = false;
	m_Toggle [ DRAIN_BARRIER ] = false;
	m_Toggle[ USE_CUDA] = true;
	m_Param [ SPH_INTSTIFF ] = 1.00;
	m_Param [ SPH_VISC ] = 0.2;
	m_Param [ SPH_INTSTIFF ] = 0.50;
	m_Param [ SPH_EXTSTIFF ] = 20000;
	m_Param [ SPH_SMOOTHRADIUS ] = 0.01;

	m_Vec [ POINT_GRAV_POS ].Set ( 0, 0, 50 );
	m_Vec [ PLANE_GRAV_DIR ].Set ( 0, 0, -9.8 );
}
void FluidSystem::setPoint(Fluid* a)
{
	a->sph_force.Set(0,0,0);
	a->vel.Set(0,0,0);
	a->vel_eval.Set(0,0,0);
	a->next = 0;
	a->pressure = 0;
	a->density = 0;
}/*
 int FluidSystem::AddPoint ()
 {
 xref ndx;	
 Fluid* f = (Fluid*) AddElem ( 0, ndx );	
 setPoint(f);
 return ndx;
 }

 int FluidSystem::AddPointReuse ()
 {
 xref ndx;	
 Fluid* f;
 if ( NumPoints() <= mBuf[0].max-2 )
 f = (Fluid*) AddElem ( 0, ndx );
 else
 f = (Fluid*) RandomElem ( 0, ndx );

 setPoint(f);
 return ndx;
 }
 */
void FluidSystem::Run ()
{

	float ss = m_Param [ SPH_PDIST ] / m_Param[ SPH_SIMSCALE ];		// simulation scale (not Schutzstaffel)
	Grid_InsertParticles ();



	if ( m_Toggle[USE_CUDA] ) {




		TransferToCUDA ( mBuf[0].data, NumPoints() );

		SPH_ComputePressureCUDA ();

        //float ** out_matrix;
		//float3 * out_mean_p;
  
	//allocateCUDAspace(&out_matrix, &out_mean_p);
		//freeCUDAspace(out_matrix, out_mean_p);

        SPH_ComputeForceCUDA ();
			m_Time += m_DT;

        SPH_ComputeDensity();
        
        SPH_ComputeVertexNumber();


		triangle_number=SPH_ExclusiveScan();

	


		if(triangle_number>max_triangle_number)
		{
			delete points_List;
			points_List=new float3[triangle_number];
			max_triangle_number=triangle_number;

			cudaFree(pointsList);
			cudaMalloc((void **) & pointsList, triangle_number*sizeof(float3));

		}

		SPH_MarchingCube();
		

      //  ExclusiveScan(vnumber,number);
		SPH_AdvanceCUDA( m_DT, m_DT/m_Param[SPH_SIMSCALE],m_Time );
		

		TransferFromCUDA ( mBuf[0].data,  NumPoints() );
		TransferFromCUDATriangles(points_List, triangle_number);


		//				Advance();

	} else {


		SPH_ComputePressure ();
		SPH_ComputeForce ();		
		Advance();
	}		


}



void FluidSystem::SPH_DrawDomain ()
{
	Vector3DF min, max;
	min = m_Vec[SPH_VOLMIN];
	max = m_Vec[SPH_VOLMAX];
	min.z += 0.5;

	glColor3f ( 0.0, 0.0, 1.0 );
	glBegin ( GL_LINES );
	glVertex3f ( min.x, min.y, min.z );	glVertex3f ( max.x, min.y, min.z );
	glVertex3f ( min.x, max.y, min.z );	glVertex3f ( max.x, max.y, min.z );
	glVertex3f ( min.x, min.y, min.z );	glVertex3f ( min.x, max.y, min.z );
	glVertex3f ( max.x, min.y, min.z );	glVertex3f ( max.x, max.y, min.z );
	glEnd ();
}

void FluidSystem::Advance ()
{
	char *ptnData_start;
	Fluid* pnt;
	Vector3DF norm, acc, vnext, min, max;

	double adj;
	float speedLimit, speedLimit2, simScale, particleRadius;
	float stiff, damp, speed, diff; 
	speedLimit = m_Param[SPH_LIMIT];
	speedLimit2 = speedLimit*speedLimit;

	stiff = m_Param[SPH_EXTSTIFF];
	damp = m_Param[SPH_EXTDAMP];
	particleRadius = m_Param[SPH_PRADIUS];
	min = m_Vec[SPH_VOLMIN];
	max = m_Vec[SPH_VOLMAX];
	simScale = m_Param[SPH_SIMSCALE];
	int numpnts = NumPoints();
	//#pragma omp parallel private(ptnData_start, pnt,norm, acc,vnext,adj,diff,speed)
	//	{
	//		int id = omp_get_num_threads();

	//for ( ptnData_start = mBuf[0].data; ptnData_start < ptnData_end; ptnData_start += mBuf[0].stride ) {
	for(int i = 0; i<numpnts;i ++){
		ptnData_start = mBuf[0].data + i * mBuf[0].stride;
		pnt = (Fluid*) ptnData_start;		

		// Compute acceration		
		acc = pnt->sph_force;
		acc *= m_Param[SPH_PMASS];
		//	acc *= (1/m_Param[SPH_PMASS]);

		// Velocity limiting 
		speed = acc.x*acc.x + acc.y*acc.y + acc.z*acc.z;
		speed = acc.Dot(acc);
		if ( speed > speedLimit2 ) {
			acc *= speedLimit / sqrt(speed);

		}		

		// Boundary Conditions

		// Z-axis walls
		diff = 2 * particleRadius - ( pnt->pos.z - min.z - (pnt->pos.x - m_Vec[SPH_VOLMIN].x) * m_Param[BOUND_ZMIN_SLOPE] )*simScale;
		if (diff > COLLI_DETECT ) {			
			norm.Set ( -m_Param[BOUND_ZMIN_SLOPE], 0, 1.0 - m_Param[BOUND_ZMIN_SLOPE] );
			adj = stiff * diff - damp * norm.Dot ( pnt->vel_eval );
			acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}		

		diff = 2 * particleRadius - ( max.z - pnt->pos.z )*simScale;
		if (diff > COLLI_DETECT) {
			norm.Set ( 0, 0, -1 );
			adj = stiff * diff - damp * norm.Dot ( pnt->vel_eval );
			acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}

		// X-axis walls
		if ( !m_Toggle[WRAP_X] ) {
			diff = 2 * particleRadius - ( pnt->pos.x - min.x + (sin(m_Time*10.0)-1+(pnt->pos.y*0.025)*0.25) * m_Param[FORCE_XMIN_SIN] )*simScale;	
			//diff = 2 * particleRadius - ( pnt->pos.x - min.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*simScale;	
			if (diff > COLLI_DETECT ) {
				norm.Set ( 1.0, 0, 0 );
				adj = (m_Param[ FORCE_XMIN_SIN ] + 1) * stiff * diff - damp * norm.Dot ( pnt->vel_eval ) ;
				acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;					
			}

			diff = 2 * particleRadius - ( max.x - pnt->pos.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMAX_SIN] )*simScale;	
			if (diff > COLLI_DETECT) {
				norm.Set ( -1, 0, 0 );
				adj = (m_Param[ FORCE_XMAX_SIN ]+1) * stiff * diff - damp * norm.Dot ( pnt->vel_eval );
				acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
			}
		}

		// Y-axis walls
		diff = 2 * particleRadius - ( pnt->pos.y - min.y )*simScale;			
		if (diff > COLLI_DETECT) {
			norm.Set ( 0, 1, 0 );
			adj = stiff * diff - damp * norm.Dot ( pnt->vel_eval );
			acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}
		diff = 2 * particleRadius - ( max.y - pnt->pos.y )*simScale;
		if (diff > COLLI_DETECT) {
			norm.Set ( 0, -1, 0 );
			adj = stiff * diff - damp * norm.Dot ( pnt->vel_eval );
			acc.x += adj * norm.x; acc.y += adj * norm.y; acc.z += adj * norm.z;
		}

		// Plane gravity
		if ( m_Param[PLANE_GRAV] > 0) 
			acc += m_Vec[PLANE_GRAV_DIR];



		// Leapfrog Integration ----------------------------
		vnext = acc;							
		vnext *= m_DT;
		vnext += pnt->vel;						// v(t+1/2) = v(t-1/2) + a(t) dt
		pnt->vel_eval = pnt->vel;
		pnt->vel_eval += vnext;
		pnt->vel_eval *= 0.5;					// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
		pnt->vel = vnext;
		vnext *= m_DT/simScale;
		pnt->pos += vnext;						// p(t+1) = p(t) + v(t+1/2) dt

		if ( m_Param[CLR_MODE]==1.0 ) {
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


		// Euler integration -------------------------------
		/* acc += m_Gravity;
		acc *= m_DT;
		pnt->vel += acc;				// v(t+1) = v(t) + a(t) dt
		pnt->vel_eval += acc;
		pnt->vel_eval *= m_DT/d;
		pnt->pos += pnt->vel_eval;
		pnt->vel_eval = pnt->vel;  */	


		if ( m_Toggle[WRAP_X] ) {
			diff = pnt->pos.x - (m_Vec[SPH_VOLMIN].x + 2);			// -- Simulates object in center of flow
			if ( diff <= 0 ) {
				pnt->pos.x = (m_Vec[SPH_VOLMAX].x - 2) + diff*2;				
				pnt->pos.z = 10;
			}
		}	
		//		}
	}
	m_Time += m_DT;
}

//------------------------------------------------------ SPH Setup 
//
//  Range = +/- 10.0 * 0.006 (r) =	   0.12			m (= 120 mm = 4.7 inch)
//  Container Volume (Vc) =			   0.001728		m^3
//  Rest Density (D) =				1000.0			kg / m^3
//  Particle Mass (Pm) =			   0.00020543	kg						(mass = vol * density)
//  Number of Particles (N) =		4000.0
//  Water Mass (M) =				   0.821		kg (= 821 grams)
//  Water Volume (V) =				   0.000821     m^3 (= 3.4 cups, .21 gals)
//  Smoothing Radius (R) =             0.02			m (= 20 mm = ~3/4 inch)
//  Particle Radius (Pr) =			   0.00366		m (= 4 mm  = ~1/8 inch)
//  Particle Volume (Pv) =			   2.054e-7		m^3	(= .268 milliliters)
//  Rest Distance (Pd) =			   0.0059		m
//
//  Given: D, Pm, N
//    Pv = Pm / D			0.00020543 kg / 1000 kg/m^3 = 2.054e-7 m^3	
//    Pv = 4/3*pi*Pr^3    cuberoot( 2.054e-7 m^3 * 3/(4pi) ) = 0.00366 m
//     M = Pm * N			0.00020543 kg * 4000.0 = 0.821 kg		
//     V =  M / D              0.821 kg / 1000 kg/m^3 = 0.000821 m^3
//     V = Pv * N			 2.054e-7 m^3 * 4000 = 0.000821 m^3
//    Pd = cuberoot(Pm/D)    cuberoot(0.00020543/1000) = 0.0059 m 
//
// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)

void FluidSystem::SPH_Setup ()
{
	m_Param [ SPH_SIMSCALE ] =		0.004;			// unit size
	m_Param [ SPH_VISC ] =			0.2;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
	m_Param [ SPH_RESTDENSITY ] =	600.0;			// kg / m^3
	m_Param [ SPH_PMASS ] =			0.00020543;		// kg
	m_Param [ SPH_PRADIUS ] =		0.004;			// m
	m_Param [ SPH_PDIST ] =			0.0059;			// m
	m_Param [ SPH_SMOOTHRADIUS ] =	0.005;			// m 
	m_Param [ SPH_INTSTIFF ] =		1.00;
	m_Param [ SPH_EXTSTIFF ] =		10000.0;
	m_Param [ SPH_EXTDAMP ] =		256.0;
	m_Param [ SPH_LIMIT ] =			200.0;			// m / s

	m_Toggle [ SPH_GRID ] =		false;
	m_Toggle [ SPH_DEBUG ] =	false;

	SPH_ComputeKernels ();
}

void FluidSystem::SPH_ComputeKernels ()
{
	m_Param [ SPH_PDIST ] = pow ( m_Param[SPH_PMASS] / m_Param[SPH_RESTDENSITY], 1/3.0 );
	m_R2 = m_Param [SPH_SMOOTHRADIUS] * m_Param[SPH_SMOOTHRADIUS];
	m_Poly6Kern = 315.0f / (64.0f * 3.141592 * pow( m_Param[SPH_SMOOTHRADIUS], 9) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
	m_SpikyKern = -45.0f / (3.141592 * pow( m_Param[SPH_SMOOTHRADIUS], 6) );			// Laplacian of viscocity (denominator): PI h^6
	m_LapKern = 45.0f / (3.141592 * pow( m_Param[SPH_SMOOTHRADIUS], 6) );
}


void FluidSystem::SPH_Test ( int n, int nmax )
{

	Vector3DF pos;
	Vector3DF min, max;

	Reset ( nmax );

	switch ( n ) {
	case 0:		// Wave pool


		m_Vec [ SPH_VOLMIN ].Set ( -25, -20, 0 );
		m_Vec [ SPH_VOLMAX ].Set ( 25, 20, 40 );		

		m_Vec [ SPH_INITMIN ].Set ( -20, -20, -10 );
		m_Vec [ SPH_INITMAX ].Set ( 20, 20, 10 );

		m_Param [ FORCE_XMIN_SIN ] =8.0;
		m_Param [ BOUND_ZMIN_SLOPE ] = 0.05;
		m_Vec [ PLANE_GRAV_DIR ].Set ( 0.0, 0, -9.8 );
		break;

	}	

	SPH_ComputeKernels ();

	m_Param [ SPH_SIMSIZE ] = m_Param [ SPH_SIMSCALE ] * (m_Vec[SPH_VOLMAX].z - m_Vec[SPH_VOLMIN].z);
	m_Param [ SPH_PDIST ] = pow ( m_Param[SPH_PMASS] / m_Param[SPH_RESTDENSITY], 1/3.0 );	

	float ss = m_Param [ SPH_PDIST ] / m_Param[ SPH_SIMSCALE ];	
	printf ( "Spacing: %f\n", ss);
	AddVolume ( m_Vec[SPH_INITMIN], m_Vec[SPH_INITMAX], ss );	// Create the particles

	Grid_Setup ( m_Vec[SPH_VOLMIN], m_Vec[SPH_VOLMAX], m_Param[SPH_SIMSCALE], 2 * m_Param[SPH_SMOOTHRADIUS], 1.0 );												// Setup grid
	Grid_InsertParticles ();									// Insert particles


#ifdef BUILD_CUDA
	FluidClearCUDA ();
	Sleep ( 500 );
	//printf("numPoints: %d\r\n",NumPoints());
    float3 min_;
	min_.x=-25;
    min_.y=-20;
	min_.z=-0;
	float3 max_;
	max_.x=25;
	max_.y=20;
	max_.z=40;
	FluidSetupCUDA ( NumPoints(), sizeof(Fluid), min_, max_, *(float3*)& m_GridRes, *(float3*)& m_GridSize, (int) m_Vec[EMIT_RATE].x );

	Sleep ( 500 );
	//( float sim_scale, float smooth_rad, float mass, float rest, float stiff, float visc, float speedlimit, float slope, float damp, float pRadius, float xminF, float xmaxF, float gravity )

	FluidParamCUDA ( m_Param[SPH_SIMSCALE], m_Param[SPH_SMOOTHRADIUS], m_Param[SPH_PMASS], m_Param[SPH_RESTDENSITY], m_Param[SPH_INTSTIFF], m_Param[SPH_VISC], m_Param[SPH_LIMIT],m_Param[ BOUND_ZMIN_SLOPE ], m_Param[SPH_EXTDAMP],m_Param[SPH_PRADIUS], m_Param[FORCE_XMIN_SIN],m_Param[FORCE_XMAX_SIN],m_Vec[PLANE_GRAV_DIR].z );
#endif

}

void FluidSystem::SPH_DrawTriangles(float* view_mat)
{
     glEnable ( GL_NORMALIZE );	
	 glLoadMatrixf ( view_mat );
     for (int i = 0; i < triangle_number; i+=3) {  
	    glPushMatrix ();

	

	 	glColor3f ( 0.1,0.1,0.5 );

         glBegin(GL_TRIANGLES);		// Drawing Using Triangles
	     glVertex3f( points_List[i].x,  points_List[i].y,  points_List[i].z);		// Top
	    glVertex3f(points_List[i+1].x,  points_List[i+1].y,  points_List[i+1].z);		// Bottom Left
	    glVertex3f(points_List[i+2].x,  points_List[i+2].y,  points_List[i+2].z);		// Bottom Right
         glEnd();					// Finished Drawing
		glPopMatrix ();		
	 }
	

}

// Compute Pressures - Very slow yet simple. O(n^2)
void FluidSystem::SPH_ComputePressure ()
{
	char *dat1, *dat1_end;
	char *dat2, *dat2_end;
	Fluid *p, *q;
	double dx, dy, dz, sum, dsq, c;
	double d, d2, mR, mR2;
	d = m_Param[SPH_SIMSCALE];
	d2 = d*d;
	mR = m_Param[SPH_SMOOTHRADIUS];
	mR2 = mR*mR;	
	int numpnts = NumPoints();
	//dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
	dat2_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
#pragma omp parallel private(dat1,dat2,p,q,dx,dy,dz,sum,dsq,c)
	{
		int id = omp_get_thread_num();

		//for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride ) {
		for(int i = id; i< numpnts;i+=NUM_CPU_THREADS)
		{
			dat1 = mBuf[0].data + i * mBuf[0].stride;
			p = (Fluid*) dat1;

			sum = 0.0;


			//for ( dat2 = mBuf[0].data; dat2 < dat2_end; dat2 += mBuf[0].stride ) {
			for(int j=0;j<numpnts;j++){
				dat2 = mBuf[0].data + j * mBuf[0].stride;
				q = (Fluid*) dat2;

				if ( p==q ) continue;
				dx = ( p->pos.x - q->pos.x)*d;		// dist in cm
				dy = ( p->pos.y - q->pos.y)*d;
				dz = ( p->pos.z - q->pos.z)*d;
				dsq = (dx*dx + dy*dy + dz*dz);
				if ( mR2 > dsq ) {
					c =  m_R2 - dsq;
					sum += c * c * c;
				}
			}	
			p->density = sum * m_Param[SPH_PMASS] * m_Poly6Kern ;	
			p->pressure = ( p->density - m_Param[SPH_RESTDENSITY] ) * m_Param[SPH_INTSTIFF];
		}
	}
}



// Compute Forces - Very slow, but simple. O(n^2)

void FluidSystem::SPH_ComputeForce ()
{
	char *dat1, *dat1_end;
	char *dat2, *dat2_end;
	Fluid *p, *q;
	Vector3DF force;
	register double termPressure, termVelo, termDist;
	double c, r, d, sum, dsq;
	double dx, dy, dz;
	double mR, mR2, visc; 

	d = m_Param[SPH_SIMSCALE];
	mR = m_Param[SPH_SMOOTHRADIUS];
	mR2 = (mR*mR);
	visc = m_Param[SPH_VISC];
	termVelo = m_LapKern * visc;
	int numpnts = NumPoints(); 
	dat2_end = mBuf[0].data + NumPoints()*mBuf[0].stride;

	//	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
#pragma omp parallel private(dat1,dat2,p,q,force,termPressure, termDist,c,r,sum,dsq,dx,dy,dz)
	{
		int id = omp_get_thread_num();
		//for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride ) {
		for (int i = id; i< numpnts; i+=NUM_CPU_THREADS)
		{
			dat1 = mBuf[0].data + i*mBuf[0].stride;
			p = (Fluid*) dat1;

			sum = 0.0;
			force.Set ( 0, 0, 0 );



			//for ( dat2 = mBuf[0].data; dat2 < dat2_end; dat2 += mBuf[0].stride ) {

			for(int j = 0; j< numpnts;j++)
			{
				dat2 = mBuf[0].data + j* mBuf[0].stride;
				q = (Fluid*) dat2;

				if ( p == q ) continue;
				dx = ( p->pos.x - q->pos.x )*d;			// dist in cm
				dy = ( p->pos.y - q->pos.y )*d;
				dz = ( p->pos.z - q->pos.z )*d;
				dsq = (dx*dx + dy*dy + dz*dz);
				if ( mR2 > dsq ) {
					r = sqrt ( dsq );
					c = (mR - r);
					termPressure = -0.5f * c * m_SpikyKern * ( p->pressure + q->pressure) / r;
					termDist = c * (1/p->density) * (1/q->density);
					force.x += ( termPressure * dx + termVelo * (q->vel_eval.x - p->vel_eval.x) ) * termDist;
					force.y += ( termPressure * dy + termVelo * (q->vel_eval.y - p->vel_eval.y) ) * termDist;
					force.z += ( termPressure * dz + termVelo * (q->vel_eval.z - p->vel_eval.z) ) * termDist;
				}
			}

			p->sph_force = force;		
		}
	}

}

