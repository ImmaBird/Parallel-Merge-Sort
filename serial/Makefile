CC = gcc #compiler
LDC = gcc #linker
CPROGS = sort.cx
FLAGS =

all: $(CPROGS)

%.cx:%.o
	$(LDC) $(FLAGS) $< -o $@

%.o:%.c
	$(CC) $(FLAGS) -c $^ -o $@

clean:
	rm -rf *.o *.cx

docker-run:
	docker build -t parallel-sort-serial:latest .
	docker run --rm parallel-sort-serial:latest 200000000 0 1000000000
