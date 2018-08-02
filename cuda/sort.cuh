#ifndef SORT
#define SORT

void mergeSort(int *array, int size);
__global__ void gpu_mergeSort(int *d_array, int arraySize, int chunkSize);
__global__ void gpu_merge(int *d_array, int *d_temp_data, int arraySize, int chunkSize);
__device__ void insertionSort(int *array, int a, int b);
__device__ void selectionSort(int *array, int size);
void printArray(int *array, int size);
__global__ void printGPUArray(int *d_array, int size);
int *getRandomArray(int size);
int randInt(int a, int b);

#endif
