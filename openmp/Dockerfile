FROM gcc

COPY Makefile ./
COPY sort.c ./

RUN make

ENV OMP_NUM_THREADS=2

ENTRYPOINT ["./sort.cx"]
