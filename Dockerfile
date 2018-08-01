FROM nlknguyen/alpine-mpich

COPY * ./

RUN make

CMD ["mpirun", "-n", "1", "./sort.cx"]