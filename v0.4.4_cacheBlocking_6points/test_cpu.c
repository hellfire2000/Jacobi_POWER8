#include <stdio.h>
#include <stdlib.h>
#include <altivec.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#define Type double

#define NZ 1440
#define NY 620
#define NX 768
#define VS 1
#define MIC_TARGET 1
//#define THREADS_NUM 12
#define STEP 50
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
    vector double v04, v03;
    v04 = vec_splats(0.4);
    v03 = vec_splats(0.3);

    int bi,bj,bk;		
    //#pragma omp parallel for num_threads(THREADS_NUM)
    #pragma omp parallel for private(bi,bj,bk,i,j,k) shared(v04,v03)
		for(bk = 0; bk < NZ; bk+=30)
			for(bj = 0; bj < NY; bj+=62)
        for(bi = 0; bi < NX; bi+=768)

		for(k = VS; k < 30+VS; k ++)
			for(j = VS; j < 62+VS; j++)
        for(i = VS; i < 768+VS; i+=12){

        //output
          vector double r0, r1, r2, r3, r4, r5;
        //source
          vector double n1[6], s1[6], t1[6], b1[6], cur[7];
        //temp
          vector double vtmp[6], vtmp2[6]; // short for vtmp[1][4]

        //value assignment
          cur[0] = vec_xld2( -8, &src[bk+k][bj+j][bi+i]);
          cur[1] = vec_xld2(  8, &src[bk+k][bj+j][bi+i]);
          cur[2] = vec_xld2( 24, &src[bk+k][bj+j][bi+i]);
          cur[3] = vec_xld2( 40, &src[bk+k][bj+j][bi+i]);
          cur[4] = vec_xld2( 56, &src[bk+k][bj+j][bi+i]);
          cur[5] = vec_xld2( 72, &src[bk+k][bj+j][bi+i]);
          cur[6] = vec_xld2( 88, &src[bk+k][bj+j][bi+i]);

          n1[0] = vec_xld2(0, &src[bk+k][bj+j+1][bi+i]);
          n1[1] = vec_xld2(16, &src[bk+k][bj+j+1][bi+i]);
          n1[2] = vec_xld2(32, &src[bk+k][bj+j+1][bi+i]);
          n1[3] = vec_xld2(48, &src[bk+k][bj+j+1][bi+i]);
          n1[4] = vec_xld2(64, &src[bk+k][bj+j+1][bi+i]);
          n1[5] = vec_xld2(80, &src[bk+k][bj+j+1][bi+i]);
	
          r0 = vec_sldw(cur[0], cur[1], 2);
          r1 = vec_sldw(cur[1], cur[2], 2);
          r2 = vec_sldw(cur[2], cur[3], 2);
          r3 = vec_sldw(cur[3], cur[4], 2);
          r4 = vec_sldw(cur[4], cur[5], 2);
          r5 = vec_sldw(cur[5], cur[6], 2);

          s1[0] = vec_xld2(0, &src[bk+k][bj+j-1][bi+i]);
          s1[1] = vec_xld2(16, &src[bk+k][bj+j-1][bi+i]);
          s1[2] = vec_xld2(32, &src[bk+k][bj+j-1][bi+i]);
          s1[3] = vec_xld2(48, &src[bk+k][bj+j-1][bi+i]);
          s1[4] = vec_xld2(64, &src[bk+k][bj+j-1][bi+i]);
          s1[5] = vec_xld2(80, &src[bk+k][bj+j-1][bi+i]);

          r0 = vec_mul(v04, r0);
          r1 = vec_mul(v04, r1);
          r2 = vec_mul(v04, r2);
          r3 = vec_mul(v04, r3);
          r4 = vec_mul(v04, r4);
          r5 = vec_mul(v04, r5);

          vtmp[0] = vec_add(n1[0], s1[0]);
          vtmp[1] = vec_add(n1[1], s1[1]);
          vtmp[2] = vec_add(n1[2], s1[2]);
          vtmp[3] = vec_add(n1[3], s1[3]);
          vtmp[4] = vec_add(n1[4], s1[4]);
          vtmp[5] = vec_add(n1[5], s1[5]);


          t1[0] = vec_xld2(0, &src[bk+k+1][bj+j][bi+i]);
          t1[1] = vec_xld2(16, &src[bk+k+1][bj+j][bi+i]);
          t1[2] = vec_xld2(32, &src[bk+k+1][bj+j][bi+i]);
          t1[3] = vec_xld2(48, &src[bk+k+1][bj+j][bi+i]);
          t1[4] = vec_xld2(64, &src[bk+k+1][bj+j][bi+i]);
          t1[5] = vec_xld2(80, &src[bk+k+1][bj+j][bi+i]);
	
          r0 = vec_madd(v03, vtmp[0], r0);
          r1 = vec_madd(v03, vtmp[1], r1);
          r2 = vec_madd(v03, vtmp[2], r2);
          r3 = vec_madd(v03, vtmp[3], r3);
          r4 = vec_madd(v03, vtmp[4], r4);
          r5 = vec_madd(v03, vtmp[5], r5);

          b1[0] = vec_xld2(0, &src[bk+k-1][bj+j][bi+i]);
          b1[1] = vec_xld2(16, &src[bk+k-1][bj+j][bi+i]);
          b1[2] = vec_xld2(32, &src[bk+k-1][bj+j][bi+i]);
          b1[3] = vec_xld2(48, &src[bk+k-1][bj+j][bi+i]);
          b1[4] = vec_xld2(64, &src[bk+k-1][bj+j][bi+i]);
          b1[5] = vec_xld2(80, &src[bk+k-1][bj+j][bi+i]);

          vtmp2[0] = vec_add(cur[0], cur[1]);
          vtmp2[1] = vec_add(cur[1], cur[2]);
          vtmp2[2] = vec_add(cur[2], cur[3]);
          vtmp2[3] = vec_add(cur[3], cur[4]);
          vtmp2[4] = vec_add(cur[4], cur[5]);
          vtmp2[5] = vec_add(cur[5], cur[6]);

          vtmp[0] = vec_add(t1[0], b1[0]);
          vtmp[1] = vec_add(t1[1], b1[1]);
          vtmp[2] = vec_add(t1[2], b1[2]);
          vtmp[3] = vec_add(t1[3], b1[3]);
          vtmp[4] = vec_add(t1[4], b1[4]);
          vtmp[5] = vec_add(t1[5], b1[5]);

          r0 = vec_madd(v03, vtmp[0], r0);
          r1 = vec_madd(v03, vtmp[1], r1);
          r2 = vec_madd(v03, vtmp[2], r2);
          r3 = vec_madd(v03, vtmp[3], r3);
          r4 = vec_madd(v03, vtmp[4], r4);
          r5 = vec_madd(v03, vtmp[5], r5);
	
          r0 = vec_madd(v03, vtmp2[0], r0);
          r1 = vec_madd(v03, vtmp2[1], r1);
          r2 = vec_madd(v03, vtmp2[2], r2);
          r3 = vec_madd(v03, vtmp2[3], r3);
          r4 = vec_madd(v03, vtmp2[4], r4);
          r5 = vec_madd(v03, vtmp2[5], r5);

          vec_xstd2(r0, 0, &dst[bk+k][bj+j][bi+i]);
          vec_xstd2(r1, 16, &dst[bk+k][bj+j][bi+i]);
          vec_xstd2(r2, 32, &dst[bk+k][bj+j][bi+i]);
          vec_xstd2(r3, 48, &dst[bk+k][bj+j][bi+i]);
          vec_xstd2(r4, 64, &dst[bk+k][bj+j][bi+i]);
          vec_xstd2(r5, 80, &dst[bk+k][bj+j][bi+i]);
	

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
				if (fabs(dst[k][j][i] - (0.4*src[k][j][i] + 0.3*(src[k-1][j][i]+src[k+1][j][i]+src[k][j-1][i]+src[k][j+1][i]+src[k][j][i-1]+src[k][j][i+1]))) > 10e-6){
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

