FROM nlknguyen/alpine-mpich

COPY Makefile ./
COPY sort.c ./

RUN make

CMD ["mpirun", "-n", "8", "./sort.cx", "100000000", "0", "100"]
