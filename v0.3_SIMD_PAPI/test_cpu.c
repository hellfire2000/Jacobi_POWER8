#include <stdio.h>
#include <stdlib.h>
#include <altivec.h>
//#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>

#ifdef IBMOPTMODPERF
#include <papi.h>
int EventSet = PAPI_NULL;
double t_GradientAndLossBatchCalc;
int eventlist[] = {
  0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0,
  0
};
#endif 



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
			for(j = VS; j < NY+VS; j++)
        for(i = VS; i < NX+VS; i+=8){

        //output
          vector double r0, r1, r2, r3; 
        //source
          vector double n1[4], s1[4], t1[4], b1[4], cur[5];
        //temp
          vector double vtmp[4]; // short for vtmp[1][4]

        //value assignment
          cur[0] = vec_xld2(0, &src[k][j][i-1]);
          cur[1] = vec_xld2(16, &src[k][j][i-1]);
          cur[2] = vec_xld2(32, &src[k][j][i-1]);
          cur[3] = vec_xld2(48, &src[k][j][i-1]);
          cur[4] = vec_xld2(64, &src[k][j][i-1]);
           
          n1[0] = vec_xld2(0, &src[k][j+1][i]);
          n1[1] = vec_xld2(16, &src[k][j+1][i]);
          n1[2] = vec_xld2(32, &src[k][j+1][i]);
          n1[3] = vec_xld2(48, &src[k][j+1][i]);

          s1[0] = vec_xld2(0, &src[k][j-1][i]);
          s1[1] = vec_xld2(16, &src[k][j-1][i]);
          s1[2] = vec_xld2(32, &src[k][j-1][i]);
          s1[3] = vec_xld2(48, &src[k][j-1][i]);

          r0 = vec_sldw(cur[0], cur[1], 2);
          r1 = vec_sldw(cur[1], cur[2], 2);
          r2 = vec_sldw(cur[2], cur[3], 2);
          r3 = vec_sldw(cur[3], cur[4], 2);

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

#ifdef IBMOPTMODPERF
  long long elapsed_us, elapsed_cyc;
  int retval;
  char const *native_name[] =
  {
    "PM_CMPLU_STALL_THRD", "PM_NTCG_ALL_FIN", "PM_GRP_CMPL", "PM_CMPLU_STALL",
#if 0
    "PM_CMPLU_STALL_LWSYNC", "PM_CMPLU_STALL_LSU", "PM_CMPLU_STALL_OTHER_CMPL", "PM_CMPLU_STALL_REJECT",
    "PM_CMPLU_STALL", "PM_CMPLU_STALL_DCACHE_MISS", "PM_CMPLU_STALL_COQ_FULL", "PM_CMPLU_STALL_ERAT_MISS",
    "PM_GCT_NOSLOT_CYC", "PM_CMPLU_STALL_STORE", "PM_CMPLU_STALL_MEM_ECC_DELAY", "PM_CMPLU_STALL_REJ_LMQ_FULL",
    "PM_DISP_HELD", "PM_CMPLU_STALL_DMISS_L2L3", "PM_CMPLU_STALL_HWSYNC", "PM_CMPLU_STALL_DMISS_L2L3_CONFLICT",
    "PM_IC_DEMAND_CYC", "PM_CMPLU_STALL_DMISS_L21_L31", "PM_CMPLU_STALL_FLUSH", "PM_CMPLU_STALL_DMISS_LMEM",
    "PM_LSU_SRQ_FULL_CYC", "PM_CMPLU_STALL_REJECT_LHS", "PM_GRP_CMPL", "PM_CMPLU_STALL_DMISS_L3MISS",
    "PM_L1_DCACHE_RELOADED_ALL", "PM_CMPLU_STALL_DMISS_REMOTE", "PM_DISP_WT", "PM_CMPLU_STALL_ST_FWD",
    "PM_GRP_MRK", "PM_CMPLU_STALL_SCALAR_LONG", "PM_FREQ_DOWN", "PM_CMPLU_STALL_SCALAR",
    "PM_MRK_ST_CMPL", "PM_CMPLU_STALL_VSU", "PM_FLUSH_COMPLETION", "PM_CMPLU_STALL_VECTOR_LONG",
    "PM_DATA_FROM_L2_NO_CONFLICT", "PM_CMPLU_STALL_VECTOR", "PM_LSU_LMQ_SRQ_EMPTY_ALL_CYC", "PM_CMPLU_STALL_LOAD_FINISH",
    "PM_DATA_FROM_L2", "PM_CMPLU_STALL_FXU", "PM_ST_MISS_L1", "PM_CMPLU_STALL_FXLONG",
    "PM_DATA_ALL_FROM_L3_NO_CONFLICT", "PM_CMPLU_STALL_BRU_CRU", "PM_L1_DCACHE_RELOAD_VALID", "PM_CMPLU_STALL_BRU",
    "PM_L3_LD_PREF", "PM_GCT_NOSLOT_IC_MISS", "PM_L1_ICACHE_RELOADED_PREF", "PM_GCT_NOSLOT_BR_MPRED_ICMISS",
    "PM_LSU0_REJECT", "PM_GCT_NOSLOT_DISP_HELD_SRQ", "PM_L3_CO_MEPF", "PM_GCT_NOSLOT_DISP_HELD_MAP",
    "PM_LSU_REJECT_LMQ_FULL", "PM_GCT_NOSLOT_DISP_HELD_ISSQ", "PM_DC_PREF_STREAM_STRIDED_CONF", "PM_GCT_NOSLOT_BR_MPRED",
    "PM_L2_TM_REQ_ABORT", "PM_CMPLU_STALL_NTCG_FLUSH", "PM_L3_SW_PREF", "PM_GCT_NOSLOT_IC_L3MISS",
    "PM_LSU_FX_FIN", "PM_L1_ICACHE_MISS", "PM_LSU_FIN", "PM_L1_ICACHE_RELOADED_ALL",
    "PM_BRU_FIN", "PM_ST_CMPL", "PM_LD_MISS_L1", "PM_MRK_LSU_FIN",
    "PM_RUN_CYC_SMT2_SPLIT_MODE", "PM_MRK_LD_MISS_L1", "PM_INST_FROM_L2_DISP_CONFLICT_LDHITST", "PM_MRK_LD_MISS_L1_CYC",
    "PM_RUN_CYC_ST_MODE", "PM_LSU_LMQ_SRQ_EMPTY_CYC", "PM_MRK_L2_RC_DONE", "PM_L3_PREF_ALL",
    "PM_LD_REF_L1", "PM_LSU_DERAT_MISS", "PM_MRK_INST_FIN", "PM_FLUSH",

    "PM_DATA_ALL_FROM_L2", "PM_DATA_ALL_FROM_MEMORY", "PM_LD_MISS_L1", "PM_DATA_ALL_FROM_L3",
    "PM_DATA_ALL_FROM_ON_CHIP_CACHE", "PM_DATA_ALL_FROM_LMEM", "PM_DATA_ALL_FROM_RMEM", "PM_DATA_ALL_FROM_OFF_CHIP_CACHE",
    "PM_DATA_ALL_FROM_LL4", "PM_DATA_ALL_FROM_RL4", "PM_DATA_ALL_FROM_DL4", "PM_DATA_ALL_FROM_DMEM",

    "PM_DATA_FROM_L2", "PM_DATA_FROM_MEMORY", "PM_LD_MISS_L1", "PM_DATA_FROM_L3",
    "PM_DATA_FROM_ON_CHIP_CACHE", "PM_DATA_FROM_LMEM", "PM_DATA_FROM_RMEM", "PM_DATA_FROM_OFF_CHIP_CACHE",
    "PM_DATA_FROM_LL4", "PM_DATA_FROM_RL4", "PM_DATA_FROM_DL4", "PM_DATA_FROM_DMEM",
#endif
    "PM_RUN_INST_CMPL", "PM_CYC",
    NULL
  };

  printf("in papi\n");
  t_GradientAndLossBatchCalc = 0.0;
  PAPI_library_init( PAPI_VER_CURRENT );
  PAPI_create_eventset( &EventSet );
  int native, j = 0;
  for ( int i = 0; native_name[i] != NULL; i++ ) {
    retval = PAPI_event_name_to_code( (char *)native_name[i], &native );
    if ( retval != PAPI_OK ) {
      printf("PAPI_event_name_to_code(%s) failed %d\n", native_name[i], retval);
    } else {
      if ( ( retval = PAPI_add_event( EventSet, native ) ) != PAPI_OK ) {
        printf("PAPI_add_events(%s) failed %d\n", native_name[i], retval);
      } else {
        eventlist[j] = native;
        j++;
      }
    }
  }

  elapsed_us = PAPI_get_real_usec(  );
  elapsed_cyc = PAPI_get_real_cyc(  );
#endif
  


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

    if (s == STEP - 1){

#ifdef IBMOPTMODPERF
  int i, retval;
  long long values[256];
  char descr[PAPI_MAX_STR_LEN];
  long long elapsed_us = PAPI_get_real_usec(  );
  retval = PAPI_start( EventSet );
  if (retval != PAPI_OK) printf("PAPI_start failed %d\n", retval);
#endif

			kernel_simd(dst, src);

#ifdef IBMOPTMODPERF
  retval = PAPI_stop( EventSet, values );
  if (retval != PAPI_OK) printf("PAPI_stop failed %d\n", retval);
  elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;
  t_GradientAndLossBatchCalc += (elapsed_us/1.E6);
  printf("Logireg::GradientAndLossBatchCalc : \t%f\n", elapsed_us/1.E6 );
  for ( i = 0; eventlist[i] != 0; i++ ) {
    PAPI_event_code_to_name( eventlist[i], descr );
    printf( "Event: %32s\tValue: %12lld\n", descr, values[i]);
  }
#endif

    }

	}
#ifdef IBMOPTMODPERF
  elapsed_cyc = PAPI_get_real_cyc(  ) - elapsed_cyc;
  elapsed_us = PAPI_get_real_usec(  ) - elapsed_us;
  printf("Master real cycles      : \t%lld\n", elapsed_cyc );
  printf("Master real sec         : \t%0.2f\n", elapsed_us/1.E6 );
  printf("Others                  : \t%0.2f\n", elapsed_us/1.E6 - t_GradientAndLossBatchCalc);
  printf("GradientAndLossBatchCalc: \t%0.2f\n", t_GradientAndLossBatchCalc);

  PAPI_shutdown();
#endif
	
  {
		check(dst);
	}
	avg_time = sum_time/STEP;

	printf("Simd Min Time: %lf\nSimd best performance is %lf Gflops\n", simd_min_time, 1.0*NX*NY*NZ*FF/simd_min_time/1000000000);
	printf("AVG Time : %lf\n AVG performance is %lf Gflops\n",avg_time,1.0*NX*NY*NZ*FF/avg_time/1000000000);
	return 0;
}

