CC = gcc #compiler
LDC = gcc #linker
CPROGS = sort.cx
FLAGS = -fopenmp

export TAU_MAKEFILE=/opt/sw/tau/x86_64/lib/Makefile.tau-pdt-openmp-opari

all: $(CPROGS)

%.cx:%.o
	$(LDC) $(FLAGS) $< -o $@

%.o:%.c
	$(CC) $(FLAGS) -c $^ -o $@

clean:
	rm -rf *.o *.cx

docker-run:
	docker build -t parallel-sort-openmp:latest .
	docker run --rm parallel-sort-openmp:latest 16000000 0 100
