CC = gcc #compiler
LDC = gcc #linker
CPROGS = sort.cx
FLAGS =

export TAU_MAKEFILE=/opt/sw/tau/x86_64/lib/Makefile.tau-pdt-openmp-opari

all: $(CPROGS)

%.cx:%.o
	$(LDC) $(FLAGS) $< -o $@

%.o:%.c
	$(CC) $(FLAGS) -c $^ -o $@

clean:
	rm -rf *.o *.cx

docker-run:
	docker build -t parallel-sort-serial:latest .
	docker run --rm parallel-sort-serial:latest 100000000 0 100
