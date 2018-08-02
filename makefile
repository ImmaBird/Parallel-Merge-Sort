CC = mpicc #compiler
LDC = mpicc #linker
CPROGS = sort.cx
FLAGS = -fopenmp

ifeq ($(OS), Windows_NT)
	DOCKER=docker.exe
else
	DOCKER=docker
endif

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
	$(DOCKER) build -t mm:latest .
	$(DOCKER) run --rm mm:latest
