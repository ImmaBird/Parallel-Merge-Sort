CC = nvcc
LDC = nvcc
RM = /bin/rm

OS := $(shell uname)

ifeq ($(OS), Darwin)
	# clang needs extra flags for omp
	FLAGS = -Xlinker -fopenmp
	LD_FLAGS = -lomp
else
	FLAGS = -fopenmp
	LD_FLAGS = -fopenmp
endif

CPROGS = sort.cx

#all rule
all: $(CPROGS)

run: all
	./$(CPROGS)

sort.cx:sort.o
	$(LDC) $(LD_FLAGS) $< -o $@

%.o:%.cu
	$(CC) $(FLAGS) -c $^ -o $@

#clean rule
clean:
	$(RM) -rf *.o *.cx *.exp *.lib
