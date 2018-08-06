#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sort.cuh"

#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE 16

// flag if the prng has been seeded
int randNotSeeded = 1;

// tests the gpu merge sort
int main(int argc, char** argv)
{
    // variables to time the sort
    double start, stop;

    // Get cmdline args
    if (argc != 4)
    {
        printf("USAGE: %s arrayLength minValue maxValue\n", argv[0]);
        return 1;
    }

    int arrayLength = atoi(argv[1]);
    int minValue = atoi(argv[2]);
    int maxValue = atoi(argv[3]);

    // the array to test our sort on
    int *data = getRandomArray(arrayLength, minValue, maxValue);

    // gets the right answer to compare too at the end
    // int *data_qsort = (int*)malloc(arrayLength*sizeof(int));
    // memcpy(data_qsort, data, arrayLength*sizeof(int));

    // Run quick sort to have an array to check against for validation
    // start = omp_get_wtime();
    // qsort(data_qsort, arrayLength, sizeof(int), comparator);
    // stop = omp_get_wtime();
    // double qsort_time = stop - start;
    

    // runs the program and times it
    start = clock();
    mergeSort(data, arrayLength);
    stop = clock();

    // Validate
    // compareArrays(data, data_qsort, arrayLength);

    // print elapsed time
    double elapsed = (stop - start) / CLOCKS_PER_SEC;
    printf("%d, %.5f\n", arrayLength, elapsed);
    // printf("qsort time: %.3fs\n", qsort_time);

    // Cleanup
    free(data);
    // free(data_qsort);
    return 0;
}

// parallel merge sort using a GPU
void mergeSort(int *h_array, int arraySize)
{
    // Make array in gpu memory
    int *d_array;
    cudaMalloc((void **)&d_array, arraySize * sizeof(int));
    cudaMemcpy(d_array, h_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // sort
    int chunkSize = CHUNK_SIZE;
    int chunks = arraySize / chunkSize + 1;
    int blocks = chunks / THREADS_PER_BLOCK + 1;
    gpu_sort<<<blocks, THREADS_PER_BLOCK>>>(d_array, arraySize, chunkSize);
    cudaDeviceSynchronize();

    // Make temp array for the merge
    int* d_temp_data;
    cudaMalloc((void **)&d_temp_data, arraySize*sizeof(int));
    do
    {
        chunkSize *= 2;
        chunks = arraySize / chunkSize + 1;
        blocks = chunks / THREADS_PER_BLOCK + 1;
        if (blocks < 8)
        {
            // CPU does the merges
            cudaMemcpy(h_array, d_array, arraySize*sizeof(int), cudaMemcpyDeviceToHost);
            cpuMerge(h_array, arraySize, chunkSize/2);
            break;
        }

        // GPU does the merges
        gpu_merge<<<blocks, THREADS_PER_BLOCK>>>(d_array, d_temp_data, arraySize, chunkSize);
        cudaDeviceSynchronize();
    }
    while(chunkSize <= arraySize);
    
    // Free GPU memory
    cudaFree(d_array);
    cudaFree(d_temp_data);
}

// sorts a bunch of small chunks from one big array
__global__ void gpu_sort(int *d_array, int size, int chunkSize)
{
    // Figure out left and right for this thread
    int a = (threadIdx.x + blockDim.x * blockIdx.x) * chunkSize;
    if (a >= size) return;

    int b = a + chunkSize;
    if (b > size) b = size;

    insertionSort(d_array, a, b);
}

// merges small sorted arrays into on big one
__global__ void gpu_merge(int *d_array, int *d_temp_array, int arraySize, int chunkSize)
{
    int pos = (threadIdx.x + blockDim.x * blockIdx.x);
    int a = pos * chunkSize;
    if (a >= arraySize) return;
    int halfChunk = chunkSize / 2;
    int m = a + halfChunk;
    if (m >= arraySize) return;
    int b = m + halfChunk;
    if (b > arraySize) b = arraySize;

    // Watch out for integer overflow
    if (a < 0 || m < 0 || b < 0) return;

    mergeArrays(d_array, d_temp_array, a, m, b);

    memcpy(d_array+a, d_temp_array+a, (b-a)*sizeof(int));
}

// serial cpu merge chunk size is the size of one sorted arrays
void cpuMerge(int *array, int size, int chunkSize)
{
    int *buffer = (int*)malloc(size*sizeof(int));
    int *data = (int*)malloc(size*sizeof(int));
    memcpy(data, array, size * sizeof(int));
    int *temp;
    int a, b, m, halfChunk;
    
    do
    {
        chunkSize *= 2;

        halfChunk = chunkSize / 2;
        for (a = 0; a < size; a += chunkSize)
        {
            m = a + halfChunk;
            if (m >= size)
            {
                memcpy(buffer+a, data+a, (size - a) * sizeof(int));
                break;
            }
            b = m + halfChunk;
            if (b > size) b = size;

            mergeArrays(data, buffer, a, m, b);
        }

        temp = buffer;
        buffer = data;
        data = temp;
    }
    while (chunkSize < size);

    memcpy(array, data, size * sizeof(int));
    free(data);
    free(buffer);
}

// Merge two sides of an array [a, m) and [m, b)
__host__ __device__ void mergeArrays(int *data, int *buffer, int a, int m, int b)
{
    int l, r, i;
    l = a;
    r = m;
    for (i = a; i < b; i++)
    {
        if (data[l] < data[r])
        {
            buffer[i] = data[l];
            l++;
            if (l == m)
            {
                while (r < b)
                {
                    i++;
                    buffer[i] = data[r];
                    r++;
                }
                break;
            }
        }
        else
        {
            buffer[i] = data[r];
            r++;
            if (r == b)
            {
                while (l < m)
                {
                    i++;
                    buffer[i] = data[l];
                    l++;
                }
                break;
            }
        }
    }
}

// sorts an array from [a,b)
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

// prints an array
__host__ __device__ void printArray(int *d_array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", d_array[i]);
    }
    printf("\n");
}

// gets an array filled with random values
int *getRandomArray(int size, int startRange, int endRange)
{
    // seed the prng if needed
    if (randNotSeeded)
    {
        srand(time(0));
        randNotSeeded = 0;
    }

    int *array = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        array[i] = randInt(startRange, endRange);
    }
    return array;
}

// gets a random int in range [a,b)
int randInt(int a, int b)
{
    return (rand() % b) + a;
}

// used by qsort for comparisons
int comparator(const void *p, const void *q)
{
    return *(const int *)p - *(const int *)q;
}

// returns true if success
int compareArrays(int *array1, int *array2, int size)
{
    for (int i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {
            printf("Broken at index:%d :(\n", i);
            return false;
        }
    }
    return true;
}
