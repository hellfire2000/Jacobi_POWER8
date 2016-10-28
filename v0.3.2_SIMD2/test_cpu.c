#include <stdio.h>
#include <stdlib.h>
#include <altivec.h>
//#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
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
    vector double v04, v03;
    v04 = vec_splats(0.4);
    v03 = vec_splats(0.3);
		
    #pragma omp parallel for private(i,j,k)
    //#pragma omp parallel for num_threads(THREADS_NUM)
		for(k = VS; k < NZ+VS; k ++)
			for(j = VS; j < NY+VS; j+=4)
        for(i = VS; i < NX+VS; i+=2){

        //output
          vector double res0, res1, res2, res3; 
        //source
          vector double north, sourth, l1[4], r1[4], t1[4], b1[4];
        //temp
          vector double vtmp[4]; // short for vtmp[1][4]

        //value assignment

          north = vec_xld2(0, &src[k][j-1][i]);
          sourth = vec_xld2(0, &src[k][j+4][i]);

          l1[0] = vec_xld2(0, &src[k][j  ][i-1]);
          r1[0] = vec_xld2(16, &src[k][j  ][i-1]);

          l1[1] = vec_xld2(0, &src[k][j+1][i-1]);
          r1[1] = vec_xld2(16, &src[k][j+1][i-1]);
          
          l1[2] = vec_xld2(0, &src[k][j+2][i-1]);
          r1[2] = vec_xld2(16, &src[k][j+2][i-1]);
          
          l1[3] = vec_xld2(0, &src[k][j+3][i-1]);
          r1[3] = vec_xld2(16, &src[k][j+3][i-1]);

          
          t1[0] = vec_xld2(0, &src[k+1][j][i]);
          t1[1] = vec_xld2(0, &src[k+1][j+1][i]);
          t1[2] = vec_xld2(0, &src[k+1][j+2][i]);
          t1[3] = vec_xld2(0, &src[k+1][j+3][i]);


          b1[0] = vec_xld2(0, &src[k-1][j][i]);
          b1[1] = vec_xld2(0, &src[k-1][j+1][i]);
          b1[2] = vec_xld2(0, &src[k-1][j+2][i]);
          b1[3] = vec_xld2(0, &src[k-1][j+3][i]);

          res0 = vec_sldw(l1[0], r1[0], 2);
          res1 = vec_sldw(l1[1], r1[1], 2);
          res2 = vec_sldw(l1[2], r1[2], 2);
          res3 = vec_sldw(l1[3], r1[3], 2);

        //compute
          vtmp[0] = vec_add(north, res1);
          vtmp[1] = vec_add(res0, res2);
          vtmp[2] = vec_add(res1, res3);
          vtmp[3] = vec_add(res2, sourth);

          res0 = vec_mul(v04, res0);
          res1 = vec_mul(v04, res1);
          res2 = vec_mul(v04, res2);
          res3 = vec_mul(v04, res3);

          res0 = vec_madd(v03, vtmp[0], res0);
          res1 = vec_madd(v03, vtmp[1], res1);
          res2 = vec_madd(v03, vtmp[2], res2);
          res3 = vec_madd(v03, vtmp[3], res3);

          vtmp[0] = vec_add(t1[0], b1[0]);
          vtmp[1] = vec_add(t1[1], b1[1]);
          vtmp[2] = vec_add(t1[2], b1[2]);
          vtmp[3] = vec_add(t1[3], b1[3]);

          res0 = vec_madd(v03, vtmp[0], res0);
          res1 = vec_madd(v03, vtmp[1], res1);
          res2 = vec_madd(v03, vtmp[2], res2);
          res3 = vec_madd(v03, vtmp[3], res3);

          vtmp[0] = vec_add(l1[0], b1[0]);
          vtmp[1] = vec_add(l1[1], r1[1]);
          vtmp[2] = vec_add(l1[2], r1[2]);
          vtmp[3] = vec_add(l1[3], r1[3]);

          res0 = vec_madd(v03, vtmp[0], res0);
          res1 = vec_madd(v03, vtmp[1], res1);
          res2 = vec_madd(v03, vtmp[2], res2);
          res3 = vec_madd(v03, vtmp[3], res3);

          vec_xstd2(res0, 0, &dst[k][j][i]);
          vec_xstd2(res1, 0, &dst[k][j+1][i]);
          vec_xstd2(res2, 0, &dst[k][j+2][i]);
          vec_xstd2(res3, 0, &dst[k][j+3][i]);
/*

        //value assignment
          cur[0] = vec_xld2(0, &src[k][j][i-1]);
          cur[1] = vec_xld2(16, &src[k][j][i-1]);
          cur[2] = vec_xld2(32, &src[k][j][i-1]);
          cur[3] = vec_xld2(48, &src[k][j][i-1]);
          cur[4] = vec_xld2(64, &src[k][j][i-1]);

          r0 = vec_sldw(cur[0], cur[1], 2);
          r1 = vec_sldw(cur[1], cur[2], 2);
          r2 = vec_sldw(cur[2], cur[3], 2);
          r3 = vec_sldw(cur[3], cur[4], 2);
           
          n1[0] = vec_xld2(0, &src[k][j+1][i]);
          n1[1] = vec_xld2(16, &src[k][j+1][i]);
          n1[2] = vec_xld2(32, &src[k][j+1][i]);
          n1[3] = vec_xld2(48, &src[k][j+1][i]);


          s1[0] = vec_xld2(0, &src[k][j-1][i]);
          s1[1] = vec_xld2(16, &src[k][j-1][i]);
          s1[2] = vec_xld2(32, &src[k][j-1][i]);
          s1[3] = vec_xld2(48, &src[k][j-1][i]);


          t1[0] = vec_xld2(0, &src[k+1][j][i]);
          t1[1] = vec_xld2(16, &src[k+1][j][i]);
          t1[2] = vec_xld2(32, &src[k+1][j][i]);
          t1[3] = vec_xld2(48, &src[k+1][j][i]);


          b1[0] = vec_xld2(0, &src[k-1][j][i]);
          b1[1] = vec_xld2(16, &src[k-1][j][i]);
          b1[2] = vec_xld2(32, &src[k-1][j][i]);
          b1[3] = vec_xld2(48, &src[k-1][j][i]);

        //compute
          r0 = vec_mul(v04, r0);
          r1 = vec_mul(v04, r1);
          r2 = vec_mul(v04, r2);
          r3 = vec_mul(v04, r3);

        
          vtmp[0] = vec_add(n1[0], s1[0]);
          vtmp[1] = vec_add(n1[1], s1[1]);
          vtmp[2] = vec_add(n1[2], s1[2]);
          vtmp[3] = vec_add(n1[3], s1[3]);

          r0 = vec_madd(v03, vtmp[0], r0);
          r1 = vec_madd(v03, vtmp[1], r1);
          r2 = vec_madd(v03, vtmp[2], r2);
          r3 = vec_madd(v03, vtmp[3], r3);

          vtmp[0] = vec_add(t1[0], b1[0]);
          vtmp[1] = vec_add(t1[1], b1[1]);
          vtmp[2] = vec_add(t1[2], b1[2]);
          vtmp[3] = vec_add(t1[3], b1[3]);

          r0 = vec_madd(v03, vtmp[0], r0);
          r1 = vec_madd(v03, vtmp[1], r1);
          r2 = vec_madd(v03, vtmp[2], r2);
          r3 = vec_madd(v03, vtmp[3], r3);

          vtmp[0] = vec_add(cur[0], cur[1]);
          vtmp[1] = vec_add(cur[1], cur[2]);
          vtmp[2] = vec_add(cur[2], cur[3]);
          vtmp[3] = vec_add(cur[3], cur[4]);

          r0 = vec_madd(v03, vtmp[0], r0);
          r1 = vec_madd(v03, vtmp[1], r1);
          r2 = vec_madd(v03, vtmp[2], r2);
          r3 = vec_madd(v03, vtmp[3], r3);

          vec_xstd2(r0, 0, &dst[k][j][i]);
          vec_xstd2(r1, 16, &dst[k][j][i]);
          vec_xstd2(r2, 32, &dst[k][j][i]);
          vec_xstd2(r3, 48, &dst[k][j][i]);
*/
        }

}

void check(Type dst[NZ+2*VS][NY+2*VS][NX+2*VS])
{
	printf("=======================check begins===========================\n");
	int i,j,k;
	int flag = 1;
  //#pragma omp parallel for private(i,j,k)
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


  //kernel_simd(dst, src);
  //kernel_simd(dst, src);
	
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

