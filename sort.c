#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int *mergeSort(int *array, int size);
void recMergeSort(int *ans, int *array, int a, int b);
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

int *mergeSort(int *array, int size)
{
    int *ans = (int *)malloc(size * sizeof(int));
    recMergeSort(ans, array, 0, size);
    return ans;
}

void recMergeSort(int *ans, int *array, int a, int b)
{
    // printf("Start a:%d b:%d size:%d\n", a, b, b - a);
    // printf("array: ");
    // printArray(array, arrayLength);
    // printf("ans:   ");
    // printArray(ans, arrayLength);
    // fflush(stdout);
    int size = b - a;
    if (b - a > 3)
    {
        int m = size / 2 + a;
        recMergeSort(ans, array, a, m);
        recMergeSort(ans, array, m, b);

        // merge
        int l = a;
        int r = m;
        for (int i = a; i < b; i++)
        {
            if (array[l] < array[r])
            {
                ans[i] = array[l];
                l++;
                if (l == m)
                {
                    while (r < b)
                    {
                        i++;
                        ans[i] = array[r];
                        r++;
                    }
                    break;
                }
            }
            else
            {
                ans[i] = array[r];
                r++;
                if (r == b)
                {
                    while (l < m)
                    {
                        i++;
                        ans[i] = array[l];
                        l++;
                    }
                    break;
                }
            }
        }

        for (int i = a; i < b; i++)
        {
            array[i] = ans[i];
        }
    }
    else
    {
        insertionSort(array, a, b);
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