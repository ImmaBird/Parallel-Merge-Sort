#include <stdio.h>
#include <stdlib.h>
#include "sort.cuh"

// __global__ void add(int* a, int* b, int* c, int n) {
//     int index = threadIdx.x + blockIdx.x * blockDim.x;
//     if (index < n) c[index] = a[index] + b[index];
// }

#define N 10
#define THREADS_PER_BLOCK 512

#define STARTRANGE 0
#define ENDRANGE 100

int main()
{
    srand(time(0));

    int *data;
    int *d_data;
    int size = N * sizeof(int);

    data = getRandomArray(N);
    printArray(data, N);

    cudaMalloc((void **)&d_data, size);

    // Copy inputs to device
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    // Launch selectionSort kernel on GPU
    selectionSort<<<1, 1>>>(d_data, N);

    // Copy result back to host
    cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);

    printArray(data, N);

    // Cleanup
    free(data);
    cudaFree(d_data);

    return 0;
}

__global__ void selectionSort(int *array, int size)
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
