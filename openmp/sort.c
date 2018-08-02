#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "omp.h"

void mergeSort(int *input, int size);
void merge(int *mergeBuffer, int *input, int a, int b, int startLeft, int startRight);
void recMergeSort(int *mergeBuffer, int *input, int a, int b);
void selectionSort(int *array, int size);
void insertionSort(int *input, int a, int b);
void printArray(int *array, int size);
int isSortedArray(int *array, int start, int end);
int *getRandomArray(int size, int minValue, int maxValue);
int randInt(int a, int b);

// flag to determine if the prng has been seeded
int randNotSeeded = 1;

int main(int argc, char *argv[])
{
    // the random number array to be sorted
    int *testArray;

    // the size of the random number array to sort
    int arrayLength;

    // the range of numbers in the random number array
    int minValue, maxValue;

    if (argc != 4)
    {
        printf("USAGE: %s arrayLength minValue maxValue", argv[0]);
    }

    arrayLength = atoi(argv[1]);
    minValue = atoi(argv[2]);
    maxValue = atoi(argv[3]);

    testArray = getRandomArray(arrayLength, minValue, maxValue);

    clock_t start, stop;
    start = clock();
    mergeSort(testArray, arrayLength);
    stop = clock();
    double duration = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Ran in %.12f seconds.\n", duration);
    if (arrayLength <= 20)
        printArray(testArray, arrayLength);

    if (isSortedArray(testArray, 0, arrayLength) == 1)
        printf("Broken :(\n");

    free(testArray);
}

void mergeSort(int *input, int size)
{
    int *mergeBuffer = (int *)malloc(size * sizeof(int));
    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int chunks = numThreads;

        if (size / chunks == 0) // Not everyone gets a chunk
        {
            #pragma omp single
            recMergeSort(mergeBuffer, input, 0, size);
        }
        else
        {
            int chunkSize = size / chunks + 1;
            int myStart = threadId * chunkSize;
            int myEnd = myStart + chunkSize;
            if (myEnd > size)
                myEnd = size;
            // printf("Thread: %d myStart: %d myEnd: %d\n", threadId, myStart, myEnd);

            recMergeSort(mergeBuffer, input, myStart, myEnd);
            // if(isSortedArray(input, myStart, myEnd) == 1)
            //     printf("Thread: %d Broken :(\n", threadId);

            #pragma omp barrier
            #pragma omp single
            {
                while (chunks > 1)
                {
                    int merges = chunks / 2;
                    for (int i = 0; i < merges; i++)
                    {
                        int taskStart = i * chunkSize * 2;
                        int taskEnd = taskStart + chunkSize * 2;
                        int startRight = (taskEnd - taskStart) / 2 + taskStart;
                        if (taskEnd > size)
                            taskEnd = size;
                        // printf("taskStart: %d taskEnd: %d startL: %d startR: %d\n", taskStart, taskEnd, taskStart, startRight);
                        #pragma omp task
                        merge(mergeBuffer, input, taskStart, taskEnd, taskStart, startRight);
                    }
                    chunks = (chunks / 2) + (chunks % 2);
                    chunkSize *= 2;
                    #pragma omp taskwait
                }
            }
        }
    }
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
        if (input[l] < input[r])
        {
            // if left is smaller...
            mergeBuffer[i] = input[l];
            l++;
            if (l == startRight)
            {
                // the left half is empty
                while (r < b)
                {
                    // copy the rest of the right half to the end
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
                while (l < startRight)
                {
                    // copy the rest of the left half to the end
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
int isSortedArray(int *array, int start, int end)
{
    for (int i = start+1; i < end; i++)
        if (array[i] < array[i - 1])
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
