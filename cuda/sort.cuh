#ifndef SORT
#define SORT

__global__ void selectionSort(int *array, int size);
void printArray(int *array, int size);
int *getRandomArray(int size);
int randInt(int a, int b);

#endif
