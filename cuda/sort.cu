#include <stdio.h>
#include <stdlib.h>
#include "sort.cuh"

// __global__ void add(int* a, int* b, int* c, int n) {
//     int index = threadIdx.x + blockIdx.x * blockDim.x;
//     if (index < n) c[index] = a[index] + b[index];
// }

#define SIZE 10
#define THREADS_PER_BLOCK 512

#define STARTRANGE 0
#define ENDRANGE 100

int main()
{
    srand(time(0));

    int *data;

    data = getRandomArray(SIZE);
    printArray(data, SIZE);

    mergeSort(data, SIZE);

    printArray(data, SIZE);

    // Cleanup
    free(data);

    return 0;
}

void mergeSort(int *array, int arraySize)
{
    // Make array in gpu memory
    int *d_array;
    cudaMalloc((void **)&d_array, arraySize*sizeof(int));
    cudaMemcpy(d_array, array, arraySize*sizeof(int), cudaMemcpyHostToDevice);

    printGPUArray<<<1, 1>>>(d_array, arraySize);

    int chunkSize = 2;
    int chunks = arraySize / chunkSize;
    
    int blocks = chunks / THREADS_PER_BLOCK + 1;

    gpu_mergeSort<<<blocks, THREADS_PER_BLOCK>>>(d_array, arraySize, chunkSize);
    cudaDeviceSynchronize();
    printGPUArray<<<1, 1>>>(d_array, arraySize);

    // Make temp array for the merge
    int* d_temp_data;
    cudaMalloc((void **)&d_temp_data, arraySize);
    chunkSize *= 2;
    while(chunkSize < arraySize)
    {
        gpu_merge<<<blocks, THREADS_PER_BLOCK>>>(d_array, d_temp_data, arraySize, chunkSize);
        chunkSize *= 2;
        chunks = arraySize / chunkSize;
        blocks = chunks / THREADS_PER_BLOCK + 1;
        cudaDeviceSynchronize();
    }

    // Copy result back to host
    cudaMemcpy(array, d_array, arraySize*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

__global__ void gpu_mergeSort(int *d_array, int arraySize, int chunkSize)
{
    // Figure out left and right for this thread
    int l = (threadIdx.x + blockDim.x * blockIdx.x) * chunkSize;
    if (l >= arraySize) return;
    int r = l + chunkSize;
    if (r >= arraySize) r = arraySize - 1;

    insertionSort(d_array, l, r);
}

__global__ void gpu_merge(int *d_array, int *d_temp_array, int arraySize, int chunkSize)
{
    // Figure out left and right for this thread
    int l = (threadIdx.x + blockDim.x * blockIdx.x) * chunkSize;
    if (l >= arraySize) return;
    int r = l + chunkSize;
    if (r >= arraySize) r = arraySize - 1;

    if (threadIdx.x == 0)
    {
        printf("(Before) Block: %d l: %d r: %d\n", blockIdx.x, l, r);
        for (int i = l; i < r; i++)
        {
            printf("%d ", d_temp_array[i]);
        }
        printf("\n");
    }

    int cur_l = l;
    int m = (l + chunkSize) / 2;
    int cur_r = m;
    for (int i = l; i < r; i++) 
    {
        if (cur_r >= r || (cur_l < m && d_array[cur_l] > d_array[cur_r]))
        {
            // left bigger than right
            d_temp_array[i] = d_array[cur_l];
            cur_l++;
        }
        else
        {
            // right bigger than left
            d_temp_array[i] = d_array[cur_r];
            cur_r++;
        }     
    }

    memcpy(d_array+l, d_temp_array+l, chunkSize*sizeof(int));

    if (threadIdx.x == 0)
    {
        printf("(After) Block: %d\n", blockIdx.x);
        for (int i = l; i < r; i++)
        {
            printf("%d ", d_temp_array[i]);
        }
        printf("\n");
        for (int i = l; i < r; i++)
        {
            printf("%d ", d_temp_array[i]);
        }
        printf("\n");
    }
}

__device__ void insertionSort(int *array, int a, int b)
{
    int current;
    for (int i = a + 1; i < b; i++)
    {
        current = array[i];
        for (int j = i - 1; j >= a - 1; j--)
        {
            if (j == a - 1 || current > array[j])
            {
                array[j + 1] = current;
                break;
            }
            else
            {
                array[j + 1] = array[j];
            }
        }
    }
}

__device__ void selectionSort(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        int smallest = array[i];
        int index = i;
        for (int j = i + 1; j < size; j++)
        {
            if (array[j] < smallest)
            {
                smallest = array[j];
                index = j;
            }
        }
        array[index] = array[i];
        array[i] = smallest;
    }
}

void printArray(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

__global__ void printGPUArray(int *d_array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", d_array[i]);
    }
    printf("\n");
}

int *getRandomArray(int size)
{
    int *array = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        array[i] = randInt(STARTRANGE, ENDRANGE);
    }
    return array;
}

int randInt(int a, int b)
{
    return (rand() % b) + a;
}
