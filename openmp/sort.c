#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "omp.h"

int cmpfunc (const void * a, const void * b);
void mergeSort(int *input, int size);
void merge(int *mergeBuffer, int *input, int a, int b, int startLeft, int startRight);
void recMergeSort(int *mergeBuffer, int *input, int a, int b, int threads);
void recMergeSortSerial(int *mergeBuffer, int *input, int a, int b);
void selectionSort(int *array, int size);
void insertionSort(int *input, int a, int b);
void printArray(int *array, int size);
int isSortedArray(int *array, int *qsortedArray, int start, int end);
int *getRandomArray(int size, int minValue, int maxValue);
int randInt(int a, int b);

// flag to determine if the prng has been seeded
int randNotSeeded = 1;

int main(int argc, char *argv[])
{
    // the random number array to be sorted
    int *testArray;
    int *qsortedArray;

    // the size of the random number array to sort
    int arrayLength;

    // the range of numbers in the random number array
    int minValue, maxValue;

    if (argc != 4)
    {
        printf("USAGE: %s arrayLength minValue maxValue\n", argv[0]);
        return 1;
    }

    arrayLength = atoi(argv[1]);
    minValue = atoi(argv[2]);
    maxValue = atoi(argv[3]);

    testArray = getRandomArray(arrayLength, minValue, maxValue);
    qsortedArray = malloc(arrayLength * sizeof(int));
    memcpy(qsortedArray, testArray, arrayLength * sizeof(int));
    qsort((void*)qsortedArray, arrayLength, sizeof(int), cmpfunc);

    double start, stop;
    start = omp_get_wtime();
    mergeSort(testArray, arrayLength);
    stop = omp_get_wtime();

    int num_threads;
    #pragma omp parallel
    #pragma omp single
    num_threads = omp_get_num_threads();
    double duration = ((double)stop - start);
    printf("%d,%d,%.5f\n", num_threads, arrayLength, duration);
    if (arrayLength <= 20)
        printArray(testArray, arrayLength);

    if (isSortedArray(testArray, qsortedArray, 0, arrayLength) == 1)
        printf("Broken :(\n");

    free(testArray);
}

int cmpfunc(const void *a, const void *b)
{
    return(*(int*)a - *(int*)b);
}

void mergeSort(int *input, int size)
{
    int *mergeBuffer = (int *)malloc(size * sizeof(int));
    #pragma omp parallel
    #pragma omp single
    recMergeSort(mergeBuffer, input, 0, size, omp_get_num_threads());
    free(mergeBuffer);
}

void recMergeSort(int *mergeBuffer, int *input, int a, int b, int threads)
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

        merge(mergeBuffer, input, a, b, l, r);
    }
}

void recMergeSortSerial(int *mergeBuffer, int *input, int a, int b)
{
    int size = b - a;

    if (size > 16)
    {
        // halve
        int m = size / 2 + a;  // the midpoint of the range the larger half goes to the right half

        recMergeSortSerial(mergeBuffer, input, a, m); // left half
        recMergeSortSerial(mergeBuffer, input, m, b); // right half

        // merge
        int l = a; // left begining index
        int r = m; // right begining index

        merge(mergeBuffer, input, a, b, l, r);
    }
    else
    {
        insertionSort(input, a, b);
    }
}

void merge(int *mergeBuffer, int *input, int a, int b, int startLeft, int startRight)
{
    int l = startLeft;
    int r = startRight;

    // loop over the range
    for (int i = a; i < b; i++)
    {
        if (r >= b || (l < startRight && input[l] < input[r]))
        {
            mergeBuffer[i] = input[l];
            l++;
        }
        else
        {
            mergeBuffer[i] = input[r];
            r++;
        }
    }

    // copy the sorted range back over to the input array for the next merge
    // memcpy size elements start from a
    int size = b - a;
    memcpy(input + a, mergeBuffer + a, size * sizeof(int));
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

// Returns 0 if sorted, 1 if not sorted
int isSortedArray(int *array, int *qsortedArray, int start, int end)
{
    for (int i = start; i < end; i++)
        if (array[i] != qsortedArray[i])
            return 1;
    return 0;
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
