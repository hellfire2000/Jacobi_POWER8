#include <stdio.h>
#include <stdlib.h>
#include <altivec.h>
#include <omp.h>
#include <math.h>
#define Type double

#define NZ 48
#define NY 64
#define NX 128
#define VS 4
#define MIC_TARGET 1
#define THREADS_NUM 8
#define STEP 50
#define TIME(a,b) (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))
#define TPC 4
#define L0_STRIDE 8
#define L1_STRIDE 16
#define FF 15			//flops in this stencil
Type src[NZ+2*VS][NY+2*VS][NX+2*VS];
Type dst[NZ+2*VS][NY+2*VS][NX+2*VS];

void init(Type src[NZ+2*VS][NY+2*VS][NX+2*VS], Type dst[NZ+2*VS][NY+2*VS][NX+2*VS])
{
		int i,j,k;
		//#pragma omp parallel for num_threads(THREADS_NUM)
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
		vector double v04, v03, v02;

		v04 = vec_splats(0.4);
		v03 = vec_splats(0.3);
		v02 = vec_splats(0.2);

		//#pragma omp parallel for num_threads(THREADS_NUM)
		for(k = VS; k < NZ+VS; k ++){
			for(j = VS; j < NY+VS; j ++){
				//#pragma vector nontemporal
				//#pragma noprefetch
				//#pragma novector
				//#pragma simd
				for(i = VS; i < NX+VS; i+=6){
					
				//declaration
					vector double r0, r1, r2;
					//2*north[3], 2*south[3], 2*top[3], 2*bottom[3], current[3], west, east
					vector double n2[3], n1[3], s2[3], s1[3], cur[3], west, east;
					vector double t2[3], t1[3], b2[3], b1[3];
					vector double vtmp11, vtmp12, vtmp13;
					vector double vtmp21, vtmp22, vtmp23;
					vector double vtmp31, vtmp32, vtmp33;
					
				//value assignment
					west   = (vector double){src[k][j][i-2], src[k][j][i-1]};
					cur[0] = (vector double){src[k][j][i],   src[k][j][i+1]};
					cur[1] = (vector double){src[k][j][i+2], src[k][j][i+3]};
					cur[2] = (vector double){src[k][j][i+4], src[k][j][i+5]};
					east   = (vector double){src[k][j][i+6], src[k][j][i+7]};
					
					n2[0] = (vector double){src[k][j+2][i],   src[k][j+2][i+1]};
					n2[1] = (vector double){src[k][j+2][i+2], src[k][j+2][i+3]};
					n2[2] = (vector double){src[k][j+2][i+4], src[k][j+2][i+5]};
					
					n1[0] = (vector double){src[k][j+1][i],   src[k][j+1][i+1]};
					n1[1] = (vector double){src[k][j+1][i+2], src[k][j+1][i+3]};
					n1[2] = (vector double){src[k][j+1][i+4], src[k][j+1][i+5]};		

					s1[0] = (vector double){src[k][j-1][i],   src[k][j-1][i+1]};
					s1[1] = (vector double){src[k][j-1][i+2], src[k][j-1][i+3]};
					s1[2] = (vector double){src[k][j-1][i+4], src[k][j-1][i+5]};						
							
					s2[0] = (vector double){src[k][j-2][i],   src[k][j-2][i+1]};
					s2[1] = (vector double){src[k][j-2][i+2], src[k][j-2][i+3]};
					s2[2] = (vector double){src[k][j-2][i+4], src[k][j-2][i+5]};

					t2[0] = (vector double){src[k+2][j][i],   src[k+2][j][i+1]};
					t2[1] = (vector double){src[k+2][j][i+2], src[k+2][j][i+3]};
					t2[2] = (vector double){src[k+2][j][i+4], src[k+2][j][i+5]};
					
					t1[0] = (vector double){src[k+1][j][i],   src[k+1][j][i+1]};
					t1[1] = (vector double){src[k+1][j][i+2], src[k+1][j][i+3]};
					t1[2] = (vector double){src[k+1][j][i+4], src[k+1][j][i+5]};

					b1[0] = (vector double){src[k-1][j][i],   src[k-1][j][i+1]};
					b1[1] = (vector double){src[k-1][j][i+2], src[k-1][j][i+3]};
					b1[2] = (vector double){src[k-1][j][i+4], src[k-1][j][i+5]};
					
					b2[0] = (vector double){src[k-2][j][i],   src[k-2][j][i+1]};
					b2[1] = (vector double){src[k-2][j][i+2], src[k-2][j][i+3]};
					b2[2] = (vector double){src[k-2][j][i+4], src[k-2][j][i+5]};

				//compute
					r0 = vec_mul(v04, cur[0]);
					r1 = vec_mul(v04, cur[1]);
					r2 = vec_mul(v04, cur[2]);

					vtmp11 = vec_add(n1[0], s1[0]);
					vtmp12 = vec_add(n1[1], s1[1]);
					vtmp13 = vec_add(n1[2], s1[2]);

					vtmp21 = vec_add(n2[0], s2[0]);
					vtmp22 = vec_add(n2[1], s2[1]);
					vtmp23 = vec_add(n2[2], s2[2]);

					vtmp31 = vec_add(west,   cur[1]);
					vtmp32 = vec_add(cur[0], cur[2]);
					vtmp33 = vec_add(cur[1], east);

					west   = vec_sldw(west,   cur[0], 2);
					cur[0] = vec_sldw(cur[0], cur[1], 2);
					cur[1] = vec_sldw(cur[1], cur[2], 2);
					cur[2] = vec_sldw(cur[2], east,   2);
					
					vtmp11 = vec_add(vtmp11, t1[0]);
					vtmp12 = vec_add(vtmp12, t1[1]);
					vtmp13 = vec_add(vtmp13, t1[2]);

					vtmp21 = vec_add(vtmp21, t2[0]);
					vtmp22 = vec_add(vtmp22, t2[1]);
					vtmp23 = vec_add(vtmp23, t2[2]);

					r0 = vec_madd(v02, vtmp31, r0);
					r1 = vec_madd(v02, vtmp32, r1);
					r2 = vec_madd(v02, vtmp33, r2);

					vtmp11 = vec_add(vtmp11, b1[0]);
					vtmp12 = vec_add(vtmp12, b1[1]);
					vtmp13 = vec_add(vtmp13, b1[2]);

					vtmp21 = vec_add(vtmp21, b2[0]);
					vtmp22 = vec_add(vtmp22, b2[1]);
					vtmp23 = vec_add(vtmp23, b2[2]);

					vtmp31 = vec_add(west,   cur[0]);
					vtmp32 = vec_add(cur[0], cur[1]);
					vtmp33 = vec_add(cur[1], cur[2]);

					r0 = vec_madd(v03, vtmp11, r0);
					r1 = vec_madd(v03, vtmp12, r1);
					r2 = vec_madd(v03, vtmp13, r2);

					r0 = vec_madd(v02, vtmp21, r0);
					r1 = vec_madd(v02, vtmp22, r1);
					r2 = vec_madd(v02, vtmp23, r2);

					r0 = vec_madd(v03, vtmp31, r0);
					r1 = vec_madd(v03, vtmp32, r1);
					r2 = vec_madd(v03, vtmp32, r2);

					//dst[k][j][i] = 0.4*src[k][j][i] + 0.3*(src[k-1][j][i]+src[k+1][j][i]+src[k][j-1][i]+src[k][j+1][i]+src[k][j][i-1]+src[k][j][i+1]) + 0.2*(src[k-2][j][i]+src[k+2][j][i]+src[k][j-2][i]+src[k][j+2][i]+src[k][j][i-2]+src[k][j][i+2]) ;
					vec_xstd2(r0,  0, &dst[k][j][i]);
					vec_xstd2(r1, 16, &dst[k][j][i]);
					vec_xstd2(r2, 32, &dst[k][j][i]);
					

				}
			}
		}
}

void check(Type dst[NZ+2*VS][NY+2*VS][NX+2*VS])
{
	printf("=======================check begins===========================\n");
	int i,j,k;
	int flag = 1;
	for(k = VS; k < NZ+VS; k ++){
		for(j = VS; j < NY+VS; j ++){
			//#pragma vector nontemporal
			//#pragma noprefetch
			//#pragma novector
			//#pragma simd
			for(i = VS; i < NX+VS; i ++){
				if (abs(dst[k][j][i] - (0.4*src[k][j][i] + 0.3*(src[k-1][j][i]+src[k+1][j][i]+src[k][j-1][i]+src[k][j+1][i]+src[k][j][i-1]+src[k][j][i+1]) + 0.2*(src[k-2][j][i]+src[k+2][j][i]+src[k][j-2][i]+src[k][j+2][i]+src[k][j][i-2]+src[k][j][i+2]))) > 10e-6){
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

