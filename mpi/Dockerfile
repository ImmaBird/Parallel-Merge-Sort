FROM nlknguyen/alpine-mpich

COPY Makefile ./
COPY sort.c ./

RUN make

CMD ["mpirun", "-n", "1", "./sort.cx"]
