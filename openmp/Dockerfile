FROM gcc

COPY Makefile ./
COPY sort.c ./

RUN make

ENTRYPOINT ["./sort.cx"]
