#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void mergeSort(int *input, int size);
void recMergeSort(int *mergeBuffer, int *input, int a, int b);
void selectionSort(int *array, int size);
void insertionSort(int *array, int a, int b);
void printArray(int *array, int size);
int *getRandomArray(int size);
int randInt(int a, int b);

// the size of the array to sort
int arrayLength = 16000000;

// the range of numbers in the array
int startRange = 0;
int endRange = 100;

int main(int argc, char *argv[])
{
    srand(time(0));
    int numranks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *numbers = getRandomArray(arrayLength);
    // printArray(numbers, arrayLength);

    double start = MPI_Wtime();
    mergeSort(numbers, arrayLength);
    printf("Ran in %.12f seconds.\n", MPI_Wtime() - start);

    // printArray(numbers, arrayLength);

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
    // printf("Start a:%d b:%d size:%d\n", a, b, b - a);
    // printf("array: ");
    // printArray(array, arrayLength);
    // printf("ans:   ");
    // printArray(ans, arrayLength);
    // fflush(stdout);
    int size = b - a; // the number of elements within this range
    if (size > 3)
    {
        // halve
        int m = size / 2 + a; // the midpoint of the range the larger half goes to the right half
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
        for (int i = a; i < b; i++)
        {
            input[i] = mergeBuffer[i];
        }
    }
    else
    {
        insertionSort(input, a, b);
    }
    // printf("End a:%d b:%d size:%d\n", a, b, b - a);
    // printf("array: ");
    // printArray(array, arrayLength);
    // printf("ans:   ");
    // printArray(ans, arrayLength);
    // fflush(stdout);
}

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

void insertionSort(int *array, int a, int b)
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

int *getRandomArray(int size)
{
    int *array = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        array[i] = randInt(startRange, endRange);
    }
    return array;
}

int randInt(int a, int b)
{
    return (rand() % b) + a;
}