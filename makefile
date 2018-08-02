CC = mpicc #compiler
LDC = mpicc #linker
CPROGS = sort.cx
FLAGS = -fopenmp

all: $(CPROGS)

%.cx:%.o
	$(LDC) $(FLAGS) $< -o $@

%.o:%.c
	$(CC) $(FLAGS) -c $^ -o $@

clean:
	rm -rf *.o *.cx

cluster-run:
	qsub pbs_sort

docker-run:
	docker build -t parallel-sort:latest .
	docker run --rm parallel-sort:latest
