FROM gcc

COPY Makefile ./
COPY sort.c ./

RUN make

ENV OMP_NUM_THREADS=4

ENTRYPOINT ["./sort.cx"]
