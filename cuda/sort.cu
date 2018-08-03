#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sort.cuh"

#define SIZE 16000000
#define THREADS_PER_BLOCK 512

#define STARTRANGE 0
#define ENDRANGE 100

void insertionSortA(int *array, int a, int b);

int comparator(const void *p, const void *q) 
{
    // Get the values at given addresses
    int l = *(const int *)p;
    int r = *(const int *)q;

    return l-r;
}

int main()
{
    // TODO: Breaks if using any array size that isn't a power of 2
    // TODO: Also breaks if using an array size that is a power of 2 >= 2^20 :)
    // TODO: Make the above two todos not happen
    srand(time(0));

    int *data;
    data = getRandomArray(SIZE);
    int *test = (int*)malloc(SIZE*sizeof(int));
    memcpy(test, data, SIZE*sizeof(int));
    qsort(test, SIZE, sizeof(int), comparator);

    clock_t start, stop;
    start = clock();
    mergeSort(data, SIZE);
    stop = clock();

    printf("\n\n");
    double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);


    for (int i = 0; i < SIZE; i++) {
        if (data[i] != test[i]) {
            printf("Broken :( %d\n", i);
            break;
        }
    }
    printArray(test, 15);
    printArray(data, 15);

    // Cleanup
    free(data);

    return 0;
}

void mergeSort(int *array, int arraySize)
{
    cudaError_t err;

    // Make array in gpu memory
    int *d_array;
    err = cudaMalloc((void **)&d_array, arraySize*sizeof(int));
    //printf("malloc: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_array, array, arraySize*sizeof(int), cudaMemcpyHostToDevice);
    //printf("cpy: %s\n", cudaGetErrorString(err));

    int chunkSize = 2;
    int chunks = arraySize / chunkSize + 1;
    int blocks = chunks / THREADS_PER_BLOCK + 1;

    gpu_mergeSort<<<blocks, THREADS_PER_BLOCK>>>(d_array, arraySize, chunkSize);
    err = cudaDeviceSynchronize();
    //printf("Merge Sort: %s\n", cudaGetErrorString(err));

    // Make temp array for the merge
    int* d_temp_data;
    cudaMalloc((void **)&d_temp_data, arraySize*sizeof(int));
    while(chunkSize <= arraySize)
    {
        chunkSize *= 2;
        chunks = arraySize / chunkSize + 1;
        blocks = chunks / THREADS_PER_BLOCK + 1;
        gpu_merge<<<blocks, THREADS_PER_BLOCK>>>(d_array, d_temp_data, arraySize, chunkSize);
        err = cudaDeviceSynchronize();
        //printf("Merge: %s chunkSize: %d\n", cudaGetErrorString(err), chunkSize);
    }

    // Copy result back to host
    cudaMemcpy(array, d_array, arraySize*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

__global__ void gpu_mergeSort(int *d_array, int arraySize, int chunkSize)
{
    // Figure out left and right for this thread
    int a = (threadIdx.x + blockDim.x * blockIdx.x) * chunkSize;
    if (a >= arraySize) return;

    int b = a + chunkSize;
    if (b > arraySize) b = arraySize;

    insertionSort(d_array, a, b);
}

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

    // if (a == 0)
    // {
    //    printf("a:%d m:%d b:%d cs:%d\n", a, m, b, chunkSize);
    // }

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

void insertionSortA(int *array, int a, int b)
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
