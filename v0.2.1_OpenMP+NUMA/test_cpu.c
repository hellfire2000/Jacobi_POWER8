#include <stdio.h>
#include <stdlib.h>
//#include <immintrin.h>
#include <omp.h>
#include <math.h>
#define Type double

#define NZ 480
#define NY 640
#define NX 1280
#define VS 1
#define MIC_TARGET 1
//#define THREADS_NUM 12
#define STEP 150
#define TIME(a,b) (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))
#define TPC 4
#define L0_STRIDE 8
#define L1_STRIDE 16
#define FF 8			//flops in this stencil
Type src[NZ+2*VS][NY+2*VS][NX+2*VS];
Type dst[NZ+2*VS][NY+2*VS][NX+2*VS];

void init(Type src[NZ+2*VS][NY+2*VS][NX+2*VS], Type dst[NZ+2*VS][NY+2*VS][NX+2*VS])
{
		int i,j,k;
		//#pragma omp parallel for num_threads(THREADS_NUM)
    #pragma omp parallel for private(i,j,k)
		for(k = 0; k < NZ+2*VS; k ++)
			for(j = 0; j < NY+2*VS; j ++)
				for(i = 0; i < NX+2*VS; i ++)
				{
					src[k][j][i] = 0.001*k+0.002*i+0.003*j+sqrt(1.0*2);
					dst[k][j][i] = 0;
				}
}




void kernel_simd(Type dst[NZ+2*VS][NY+2*VS][NX+2*VS], Type src[NZ+2*VS][NY+2*VS][NX+2*VS])
{
		int i,j,k;
		
    #pragma omp parallel for private(i,j,k)
    //#pragma omp parallel for num_threads(THREADS_NUM)
		for(k = VS; k < NZ+VS; k ++)
			for(j = VS; j < NY+VS; j++)
				//#pragma vector nontemporal
				//#pragma noprefetch
				//#pragma novector
				//#pragma simd
				for(i = VS; i < NX+VS; i++)
					dst[k][j][i] = 0.4*src[k][j][i] + 0.3*(src[k-1][j][i]+src[k+1][j][i]+src[k][j-1][i]+src[k][j+1][i]+src[k][j][i-1]+src[k][j][i+1]) ;
}

void check(Type dst[NZ+2*VS][NY+2*VS][NX+2*VS])
{
	printf("=======================check begins===========================\n");
	int i,j,k;
	int flag = 1;
  #pragma omp parallel for private(i,j,k)
	for(k = VS; k < NZ+VS; k ++){
		for(j = VS; j < NY+VS; j ++){
			//#pragma vector nontemporal
			//#pragma noprefetch
			//#pragma novector
			//#pragma simd
			for(i = VS; i < NX+VS; i ++){
				if (abs(dst[k][j][i] - (0.4*src[k][j][i] + 0.3*(src[k-1][j][i]+src[k+1][j][i]+src[k][j-1][i]+src[k][j+1][i]+src[k][j][i-1]+src[k][j][i+1]))) > 10e-6){
					flag = 0;
					printf("%.16lf ", dst[k][j][i]);
				}
			}
		}
	}
	if(flag == 1){
		printf("Success!\n");
	}
	printf("========================check ends============================\n");
}

int main()
{
	double min_time = 10000;
	double simd_min_time = 10000;
	double sum_time = 0.0;
	double avg_time;
	printf("Init\n");
	{
		init(src, dst);
	}
	int s;


  kernel_simd(dst, src);
  kernel_simd(dst, src);
	
  printf("=====simd part=====\n");
	for(s = 0; s < STEP; s ++)
	{
		struct timeval t1, t2;
		gettimeofday(&t1, NULL);
		{
			kernel_simd(dst, src);
		}
		gettimeofday(&t2, NULL);
		if(TIME(t1,t2) < simd_min_time) simd_min_time = TIME(t1,t2);
		printf("TIME: %lf\n", TIME(t1,t2));
		sum_time += TIME(t1,t2);
	}
	{
		check(dst);
	}
	avg_time = sum_time/STEP;

	printf("Simd Min Time: %lf\nSimd best performance is %lf Gflops\n", simd_min_time, 1.0*NX*NY*NZ*FF/simd_min_time/1000000000);
	printf("AVG Time : %lf\n AVG performance is %lf Gflops\n",avg_time,1.0*NX*NY*NZ*FF/avg_time/1000000000);
	return 0;
}

