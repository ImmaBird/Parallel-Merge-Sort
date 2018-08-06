#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void mergeSort(int *input, int size);
void recMergeSort(int *mergeBuffer, int *input, int a, int b);
void selectionSort(int *array, int size);
void insertionSort(int *input, int a, int b);
void merge(int *array, int size, int *indexes, int numranks);
void printArray(int *array, int size);
int *getRandomArray(int size, int minValue, int maxValue);
void mergeArrays(int *data, int *buffer, int a, int m, int b);
int randInt(int a, int b);
int compareArrays(int *array1, int *array2, int size);
int comparator(const void *p, const void *q);

// flag to determine if the prng has been seeded
int randNotSeeded = 1;

int main(int argc, char *argv[])
{
    int numranks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // the random number array to be sorted
    int *testArray;

    // this ranks chunk of the values
    int *myChunk;

    // the size of the random number array to sort
    int arrayLength;

    // the range of numbers in the random number array
    int minValue, maxValue;

    // the amount of values given to each node
    int chunkSize;

    // send counts and displacements for scatterv
    int *sendCounts, *displacements;

    // the rank to send your chunk to to be merged
    int buddy;

    // rank 0 gets user input
    if (rank == 0)
    {
        if (argc != 4)
        {
            printf("USAGE: %s arrayLength minValue maxValue", argv[0]);
        }

        arrayLength = atoi(argv[1]);
        minValue = atoi(argv[2]);
        maxValue = atoi(argv[3]);
    }

    // start timer
    double start, end;
    if (rank == 0)
    {
        start = MPI_Wtime();
    }

    // rank 0 sends arrayLength to everyone
    MPI_Bcast(&arrayLength, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // rank 0 creates the random number array
    if (rank == 0)
    {
        testArray = getRandomArray(arrayLength, minValue, maxValue);
    }

    // gets the right answer to compare too at the end
    // int *data_qsort;
    // if (rank == 0)
    // {
    //     data_qsort = (int *)malloc(arrayLength * sizeof(int));
    //     memcpy(data_qsort, testArray, arrayLength * sizeof(int));
    //     mergeSort(data_qsort, arrayLength);
    // }    

    // everyone calculates sendCounts and displacements
    sendCounts = (int *)malloc(numranks * sizeof(sendCounts));
    displacements = (int *)malloc(numranks * sizeof(displacements));
    for (int i = 0; i < numranks; i++)
    {
        sendCounts[i] = arrayLength / numranks;
        if (i < arrayLength % numranks)
        {
            sendCounts[i]++;
        }

        if (i == 0)
        {
            displacements[i] = 0;
        }
        else
        {
            displacements[i] = displacements[i - 1] + sendCounts[i - 1];
        }
    }

    // allocate space for myChunk
    chunkSize = sendCounts[rank];
    myChunk = (int *)malloc(chunkSize * sizeof(int));

    // rank 0 sends chunks of values to everyone
    MPI_Scatterv(testArray, sendCounts, displacements, MPI_INT, myChunk, chunkSize, MPI_INT, 0, MPI_COMM_WORLD);

    // perform merge sort on myChunk
    mergeSort(myChunk, chunkSize);

    // collect all the sorted arrays on rank 0
    MPI_Gatherv(myChunk, chunkSize, MPI_INT, testArray, sendCounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

    //printf("rank:%d chunksize:%d\n", rank, chunkSize);

    // merge all the arrays together
    if (rank == 0)
    {
        merge(testArray, arrayLength, displacements, numranks);
        end = MPI_Wtime();
        printf("%d,%d,%.5f\n", numranks, arrayLength, end - start);
        // compareArrays(testArray, data_qsort, arrayLength);
    }

    MPI_Finalize();
}

void mergeSort(int *input, int size)
{
    int *mergeBuffer = (int *)malloc(size * sizeof(int));
    recMergeSort(mergeBuffer, input, 0, size);
    free(mergeBuffer);
}

void recMergeSort(int *mergeBuffer, int *input, int a, int b)
{
    int size = b - a; // the number of elements within this range
    if (size > 16)
    {
        // halve
        int m = size / 2 + a;                   // the midpoint of the range the larger half goes to the right half
        recMergeSort(mergeBuffer, input, a, m); // left half
        recMergeSort(mergeBuffer, input, m, b); // right half

        // merge
        int l = a; // left begining index
        int r = m; // right begining index

        // loop over the range
        for (int i = a; i < b; i++)
        {
            if (input[l] < input[r])
            {
                // if left is smaller...
                mergeBuffer[i] = input[l];
                l++;
                if (l == m)
                {
                    // the left half is empty
                    while (r < b)
                    {
                        // copy the rest of the left half to the end
                        i++;
                        mergeBuffer[i] = input[r];
                        r++;
                    }
                    break;
                }
            }
            else
            {
                // if right is smaller...
                mergeBuffer[i] = input[r];
                r++;
                if (r == b)
                {
                    // the right half is empty
                    while (l < m)
                    {
                        // copy the rest of the right half to the end
                        i++;
                        mergeBuffer[i] = input[l];
                        l++;
                    }
                    break;
                }
            }
        }

        // copy the sorted range back over to the input array for the next merge
        // memcpy size elements start from a
        memcpy(input + a, mergeBuffer + a, size * sizeof(int));
    }
    else
    {
        insertionSort(input, a, b);
    }
}

// serial cpu merge chunk size is the size of one sorted arrays
void merge(int *array, int size, int *indexes, int numranks)
{
    int *buffer = (int *)malloc(size * sizeof(int));
    int *data = (int *)malloc(size * sizeof(int));
    memcpy(data, array, size * sizeof(int));
    int *temp;
    int a, b, m, i, chunkSize, halfChunk;
    for (chunkSize = 2; chunkSize <= numranks; chunkSize *= 2)
    {
        halfChunk = chunkSize / 2;
        for (i = 0; i+halfChunk < numranks; i += chunkSize)
        {
            a = indexes[i];
            m = indexes[i+halfChunk];
            b = i+chunkSize != numranks? indexes[i+chunkSize] : size;
            //printf("a:%d m:%d b:%d\n",i,i+halfChunk, i+chunkSize);
            //printf("chunksize:%d i:%d a:%d m:%d b:%d\n",chunkSize, i, a, m, b);
            //fflush(stdout);
            mergeArrays(data, buffer, a, m, b);
        }

        temp = buffer;
        buffer = data;
        data = temp;
    }

    memcpy(array, data, size * sizeof(int));
    free(buffer);
    free(data); 
}

void mergeArrays(int *data, int *buffer, int a, int m, int b)
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

// sorts the values of an array within a range [a,b)
void insertionSort(int *input, int a, int b)
{
    int current, i, j;
    // loop over the range
    for (i = a + 1; i < b; i++)
    {
        current = input[i]; // the value to insert
        // loop back from the current value
        for (j = i - 1; j >= a - 1; j--)
        {
            // if the current value is greater than the value being checked...
            if (j == a - 1 || current > input[j])
            {
                // insert it in front of that value
                input[j + 1] = current;
                break;
            }
            else
            {
                // if its less than the value being checked
                // shift the value up the array to make room for the current value
                input[j + 1] = input[j];
            }
        }
    }
}

// selection sort UNUSED
void selectionSort(int *array, int size)
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

// prints and array of integers in a row
void printArray(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int *getRandomArray(int size, int minValue, int maxValue)
{
    // seed the prng if needed
    if (randNotSeeded)
    {
        srand(time(0));
        randNotSeeded = 0;
    }

    // allocate an array and set its values to random numbers
    int *randomValues = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        randomValues[i] = randInt(minValue, maxValue);
    }
    return randomValues;
}

// returns a random integer within a range [a,b)
int randInt(int a, int b)
{
    return (rand() % b) + a;
}

// returns true if success
int compareArrays(int *array1, int *array2, int size)
{
    for (int i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {
            printf("Broken at index:%d :(\n", i);
            return 0;
        }
    }
    return 1;
}

// used by qsort for comparisons
int comparator(const void *p, const void *q)
{
    return *(const int *)p - *(const int *)q;
}

void mergeSort(int *input, int size)
{
    int *mergeBuffer = (int *)malloc(size * sizeof(int));
    #pragma omp parallel
    #pragma omp single
    recMergeSortOmp(mergeBuffer, input, 0, size, omp_get_num_threads());
    free(mergeBuffer);
}

void recMergeSortOmp(int *mergeBuffer, int *input, int a, int b, int threads)
{
    int size = b - a; // the number of elements within this range
    if (threads == 1)
    {
        recMergeSortSerial(mergeBuffer, input, a, b);
    }
    else
    {
        // halve
        int m = size / 2 + a;  // the midpoint of the range the larger half goes to the right half
        
        #pragma omp task
        recMergeSort(mergeBuffer, input, a, m, threads/2); // left half
        #pragma omp task
        recMergeSort(mergeBuffer, input, m, b, threads-threads/2); // right half
        
        #pragma omp taskwait

        // merge
        int l = a; // left begining index
        int r = m; // right begining index

        mergeArrays(mergeBuffer, input, a, b, l, r);
    }
}