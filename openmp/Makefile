CC = gcc #compiler
LDC = gcc #linker
CPROGS = sort.cx
FLAGS = -fopenmp

all: $(CPROGS)

%.cx:%.o
	$(LDC) $(FLAGS) $< -o $@

%.o:%.c
	$(CC) $(FLAGS) -c $^ -o $@

clean:
	rm -rf *.o *.cx

docker-run:
	docker build -t parallel-sort-openmp:latest .
	docker run --rm parallel-sort-openmp:latest 200000000 0 100
