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
void printArray(int *array, int size);
int *getRandomArray(int size, int minValue, int maxValue);
int randInt(int a, int b);

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

    // rank 0 sends arrayLength to everyone
    MPI_Bcast(&arrayLength, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // rank 0 creates the random number array
    if (rank == 0)
    {
        testArray = getRandomArray(arrayLength, minValue, maxValue);
    }

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

    double start = MPI_Wtime();
    mergeSort(testArray, arrayLength);
    printf("Ran in %.12f seconds.\n", MPI_Wtime() - start);

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
