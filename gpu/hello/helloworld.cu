#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
__global__ void helloFromGPU(void)
{
        printf("Hello World from GPU!\n");
        
}

int main(void)
{
        // hello from cpu
        cudaError_t cudaStatus;
            printf("Hello World from CPU!\n");

                helloFromGPU << <1, 10 >> > ();
                    cudaDeviceReset();//重置CUDA设备释放程序占用的资源
                        system("pause");
                            return 0;
                            
}

