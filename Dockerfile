FROM nlknguyen/alpine-mpich

COPY makefile ./
COPY sort.c ./

RUN make

CMD ["mpirun", "-n", "1", "./sort.cx"]
