#pragma once
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: Error code %d: '%s'\n",__FILE__,__LINE__,e,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
