#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

#define NX 3335 // 有效数据个数
#define N 5335 // 补0之后的数据长度
#define BATCH 1
#define BLOCK_SIZE 1024
using std::cout;
using std::endl;


/**
* 功能：判断两个 cufftComplex 数组的是否相等
* 输入：idataA 输入数组A的头指针
* 输入：idataB 输出数组B的头指针
* 输入：size 数组的元素个数
* 返回：true | false
*/
bool IsEqual(cufftComplex *idataA, cufftComplex *idataB, const int size)
{
    for (int i = 0; i < size; i++)
    {
        if (abs(idataA[i].x - idataB[i].x) > 0.000001 || abs(idataA[i].y - idataB[i].y) > 0.000001)
            return false;
    }

    return true;
}



/**
* 功能：实现 cufftComplex 数组的尺度缩放，也就是乘以一个数
* 输入：idata 输入数组的头指针
* 输出：odata 输出数组的头指针
* 输入：size 数组的元素个数
* 输入：scale 缩放尺度
*/
static __global__ void cufftComplexScale(cufftComplex *idata, cufftComplex *odata, const int size, float scale)
{
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < size)
    {
        odata[threadID].x = idata[threadID].x * scale;
        odata[threadID].y = idata[threadID].y * scale;
    }
}

int main()
{
    cufftComplex *data_dev; // 设备端数据头指针
    cufftComplex *data_Host = (cufftComplex*)malloc(NX*BATCH * sizeof(cufftComplex)); // 主机端数据头指针
    cufftComplex *resultFFT = (cufftComplex*)malloc(N*BATCH * sizeof(cufftComplex)); // 正变换的结果
    cufftComplex *resultIFFT = (cufftComplex*)malloc(NX*BATCH * sizeof(cufftComplex)); // 先正变换后逆变换的结果

    // 初始数据
    for (int i = 0; i < NX; i++)
    {
        data_Host[i].x = float((rand() * rand()) % NX) / NX;
        data_Host[i].y = float((rand() * rand()) % NX) / NX;
    }


    dim3 dimBlock(BLOCK_SIZE); // 线程块
    dim3 dimGrid((NX + BLOCK_SIZE - 1) / dimBlock.x); // 线程格

    cufftHandle plan; // 创建cuFFT句柄
    cufftPlan1d(&plan, N, CUFFT_C2C, BATCH);

    // 计时
    clock_t start, stop;
    double duration;
    start = clock();

    cudaMalloc((void**)&data_dev, sizeof(cufftComplex)*N*BATCH); // 开辟设备内存
    cudaMemset(data_dev, 0, sizeof(cufftComplex)*N*BATCH); // 初始为0
    cudaMemcpy(data_dev, data_Host, NX * sizeof(cufftComplex), cudaMemcpyHostToDevice); // 从主机内存拷贝到设备内存

    cufftExecC2C(plan, data_dev, data_dev, CUFFT_FORWARD); // 执行 cuFFT，正变换
    cudaMemcpy(resultFFT, data_dev, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost); // 从设备内存拷贝到主机内存

    cufftExecC2C(plan, data_dev, data_dev, CUFFT_INVERSE); // 执行 cuFFT，逆变换
    cufftComplexScale << <dimGrid, dimBlock >> > (data_dev, data_dev, N, 1.0f / N); // 乘以系数
    cudaMemcpy(resultIFFT, data_dev, NX * sizeof(cufftComplex), cudaMemcpyDeviceToHost); // 从设备内存拷贝到主机内存

    stop = clock();
    duration = (double)(stop - start) * 1000 / CLOCKS_PER_SEC;
    cout << "时间为 " << duration << " ms" << endl;

    cufftDestroy(plan); // 销毁句柄
    cudaFree(data_dev); // 释放空间

    cout << IsEqual(data_Host, resultIFFT, NX) << endl;

    return 0;
}