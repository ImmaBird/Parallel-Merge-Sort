#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void transposeMat(int *mat, int n);
void printMat(int *mat, int rowSize, int n);

int main(int argc, char *argv[])
{
    int numranks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n;                                       // rank 0
    int *A, *B, *ans;                            // rank 0
    int *myB, *myAns;                            // all ranks
    int *sendCounts, *displacements;             // all ranks
    int sizeMyB, myNumRows;                      // all ranks
    int i, j, z, sum;                            // all ranks
    double start, end;                           // rank 0
    double startNetworkTime, elapsedNetworkTime; // rank 0
    int numThreads;                              // all ranks
    int totalNumThreads;                         // rank 0

    // rank 0 calculates n and start the timer
    if (rank == 0)
    {
        elapsedNetworkTime = 0;
        start = MPI_Wtime();
        if (argc != 2)
        {
            printf("USAGE: command mat_size");
            return -1;
        }
        else
        {
            n = atoi(argv[1]);
        }
    }

    // start netowork time
    if (rank == 0)
    {
        startNetworkTime = MPI_Wtime();
    }

    // rank 0 sends n to everyone
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // add network time to elapsed network time
    if (rank == 0)
    {
        elapsedNetworkTime += MPI_Wtime() - startNetworkTime;
    }

    // everyone allocates space for A
    A = (int *)malloc(n * n * sizeof(A));

    // rank 0 calculates A and B
    if (rank == 0)
    {
        B = (int *)malloc(n * n * sizeof(B));

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                A[i * n + j] = 2 * i + j;
                // transpose B to make sending easier and reduce cache misses
                B[j * n + i] = i + 3 * j;
            }
        }
    }

    // start netowork time
    if (rank == 0)
    {
        startNetworkTime = MPI_Wtime();
    }

    // rank 0 sends everyone A
    MPI_Bcast(A, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // add network time to elapsed network time
    if (rank == 0)
    {
        elapsedNetworkTime += MPI_Wtime() - startNetworkTime;
    }

    // everyone calculates sendCounts and displacements
    sendCounts = (int *)malloc(numranks * sizeof(sendCounts));
    displacements = (int *)malloc(numranks * sizeof(displacements));
    for (i = 0; i < numranks; i++)
    {
        sendCounts[i] = n / numranks;
        if (i < n % numranks)
        {
            sendCounts[i]++;
        }
        sendCounts[i] = sendCounts[i] * n;

        if (i == 0)
        {
            displacements[i] = 0;
        }
        else
        {
            displacements[i] = displacements[i - 1] + sendCounts[i - 1];
        }
    }

    // everyone allocates space for their chunk of B and their ans
    sizeMyB = sendCounts[rank];
    myNumRows = sizeMyB / n;
    myB = (int *)malloc(sizeMyB * sizeof(myB));

    // start netowork time
    if (rank == 0)
    {
        startNetworkTime = MPI_Wtime();
    }

    // rank 0 sends chunks of B to everyone
    MPI_Scatterv(B, sendCounts, displacements, MPI_INT, myB, sizeMyB, MPI_INT, 0, MPI_COMM_WORLD);

    // add network time to elapsed network time
    if (rank == 0)
    {
        elapsedNetworkTime += MPI_Wtime() - startNetworkTime;
    }

    // everyone prints what their arrays look like
    if (n <= 10)
    {
        for (i = 0; i < numranks; i++)
        {
            if (rank == i)
            {
                printf("Rank: %d Mat: A\n", rank);
                printMat(A, n, n * n);
                printf("Rank: %d Mat: B Chunk: %d\n", rank, sizeMyB);
                printMat(myB, n, sizeMyB);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // everyone creates their result matrix
    myAns = (int *)malloc(sizeMyB * sizeof(myAns));

// multiply matrices
#pragma omp parallel private(sum, j, z)
    {
#pragma omp single
        {
            // uncomment omp get num threads when using omp
            numThreads = 1; // omp_get_num_threads();
        }
#pragma omp for
        for (i = 0; i < myNumRows; i++)
        {
            for (j = 0; j < n; j++)
            {
                sum = 0;
                for (z = 0; z < n; z++)
                {
                    sum += A[j * n + z] * myB[i * n + z];
                }
                myAns[i * n + j] = sum;
            }
        }
    }

    // rank 0 allocates space for ans
    ans = (int *)malloc(n * n * sizeof(ans));

    // start netowork time
    if (rank == 0)
    {
        startNetworkTime = MPI_Wtime();
    }

    // rank 0 gathers all of the myAns into ans
    MPI_Gatherv(myAns, sizeMyB, MPI_INT, ans, sendCounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

    // add network time to elapsed network time
    if (rank == 0)
    {
        elapsedNetworkTime += MPI_Wtime() - startNetworkTime;
    }

    // transpose the answer because B was transposed so now ans is
    transposeMat(ans, n);

    // rank 0 prints the result
    if (rank == 0)
    {
        end = MPI_Wtime();
    }
    // get the total number of omp threads
    MPI_Reduce(&numThreads, &totalNumThreads, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        if (n <= 10)
        {
            printf("ans:\n");
            printMat(ans, n, n * n);
        }
        printf("%d, %d, %d, %f, %f\n", n, numranks, totalNumThreads / numranks, end - start, elapsedNetworkTime);
    }

    // free all the memory
    if (rank == 0)
    {
        free(B);
        free(ans);
    }
    free(A);
    free(myB);
    free(myAns);
    free(sendCounts);
    free(displacements);

    MPI_Finalize();
}

void transposeMat(int *mat, int n)
{
    int temp = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            temp = mat[j * n + i];
            mat[j * n + i] = mat[i * n + j];
            mat[i * n + j] = temp;
        }
    }
}

void printMat(int *mat, int rowSize, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", mat[i]);
        if ((i + 1) % rowSize == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
}