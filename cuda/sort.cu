#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sort.cuh"

#define SIZE 10000000
#define STARTRANGE 0
#define ENDRANGE 100

#define THREADS_PER_BLOCK 1024

// flag if the prng has been seeded
int randNotSeeded = 1;

// tests the gpu merge sort
int main()
{
    // variables to time the sort
    clock_t start, stop;

    // the array to test our sort on
    int *data = getRandomArray(SIZE);

    // print the first 15 elements of the data
    if (SIZE > 15)
    {
        printArray(data, 15);
    }
    else
    {
        printArray(data, SIZE);
    }

    // gets the right answer to compare too at the end
    int *data_qsort = (int*)malloc(SIZE*sizeof(int));
    memcpy(data_qsort, data, SIZE*sizeof(int));
    qsort(data_qsort, SIZE, sizeof(int), comparator);

    // runs the program and times it
    start = clock();
    mergeSort(data, SIZE);
    stop = clock();
    

    // print the first 15 elements of the hopefully sorted data array
    printf("\n");
    if (SIZE > 15)
    {
        printArray(data_qsort, 15);
    }
    else
    {
        printArray(data_qsort, SIZE);
    }

    // prints the first 15 elements of the sorted array
    if (SIZE > 15)
    {
        printArray(data, 15);
    }
    else
    {
        printArray(data, SIZE);
    }

    // print elapsed time
    double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n\n", elapsed);

    // Cleanup
    free(data);

    return 0;
}

// parallel merge sort using a GPU
void mergeSort(int *array, int arraySize)
{
    // Make array in gpu memory
    int *d_array;
    cudaMalloc((void **)&d_array, arraySize*sizeof(int));
    cudaMemcpy(d_array, array, arraySize*sizeof(int), cudaMemcpyHostToDevice);

    int chunkSize = 2;
    int chunks = arraySize / chunkSize + 1;
    int blocks = chunks / THREADS_PER_BLOCK + 1;
    gpu_sort<<<blocks, THREADS_PER_BLOCK>>>(d_array, arraySize, chunkSize);
    cudaDeviceSynchronize();

    // Make temp array for the merge
    cudaError_t err;
    int* d_temp_data;
    cudaMalloc((void **)&d_temp_data, arraySize*sizeof(int));
    while(chunkSize <= arraySize)
    {
        chunkSize *= 2;
        chunks = arraySize / chunkSize + 1;
        blocks = chunks / THREADS_PER_BLOCK + 1;
        gpu_merge<<<blocks, THREADS_PER_BLOCK>>>(d_array, d_temp_data, arraySize, chunkSize);
        err = cudaDeviceSynchronize();
        printf("Merge: %s chunkSize: %d\n", cudaGetErrorString(err), chunkSize);
    }

    // Copy result back to host
    cudaMemcpy(array, d_array, arraySize*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

// sorts a bunch of small chunks from one big array
__global__ void gpu_sort(int *d_array, int arraySize, int chunkSize)
{
    // Figure out left and right for this thread
    int a = (threadIdx.x + blockDim.x * blockIdx.x) * chunkSize;
    if (a >= arraySize) return;

    int b = a + chunkSize;
    if (b > arraySize) b = arraySize;

    insertionSort(d_array, a, b);
}

// merges small sorted arrays into on big one
__global__ void gpu_merge(int *d_array, int *d_temp_array, int arraySize, int chunkSize)
{
    // Figure out left and right for this thread
    //printf("threadIdx: %d, blockDim: %d, blockIdx: %d\n", threadIdx.x, blockDim.x, blockIdx.x);
    int a = (threadIdx.x + blockDim.x * blockIdx.x) * chunkSize;
    if (a >= arraySize) return;
    int b = a + chunkSize;
    int m = (b - a) / 2 + a;
    if (m >= arraySize) return;
    if (b > arraySize) b = arraySize;

    int l = a;
    int r = m;
    for (int i = a; i < b; i++)
        {
            if (d_array[l] < d_array[r])
            {
                d_temp_array[i] = d_array[l];
                l++;
                if (l == m)
                {
                    while (r < b)
                    {
                        i++;
                        d_temp_array[i] = d_array[r];
                        r++;
                    }
                    break;
                }
            }
            else
            {
                d_temp_array[i] = d_array[r];
                r++;
                if (r == b)
                {
                    while (l < m)
                    {
                        i++;
                        d_temp_array[i] = d_array[l];
                        l++;
                    }
                    break;
                }
            }
        }

    memcpy(d_array+a, d_temp_array+a, (b-a)*sizeof(int));
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
int *getRandomArray(int size)
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
        array[i] = randInt(STARTRANGE, ENDRANGE);
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
int compareArrays(int *array1, int *array2)
{
    for (int i = 0; i < SIZE; i++) {
        if (array1[i] != array2[i]) {
            printf("Broken at index:%d :(\n", i);
            return false;
        }
    }
    return true;
}