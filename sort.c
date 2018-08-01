#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void selectionSort(int *array, int size);
void printArray(int *array, int size);
int *getRandomArray(int size);
int randInt(int a, int b);

// the size of the array to sort
int arrayLength = 100;

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
    printArray(numbers, arrayLength);
    selectionSort(numbers, arrayLength);
    printArray(numbers, arrayLength);

    MPI_Finalize();
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

void insertionSort(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {

        for (int j = i + 1; j < size; j++)
        {
        }
    }
}

void printArray(int *array, int size)
{
    for(int i = 0; i < size; i++)
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