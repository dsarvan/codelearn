all: program.so matmult.so

program.so: program.o
	gcc -shared -o program.so program.o

program.o: program.c
	gcc -O3 -c -Wall -Werror -fpic program.c

matmult.so: matmult.o
	mpicc -shared -o matmult.so matmult.o

matmult.o: matmult.c
	mpicc -O3 -c -Wall -Werror -fpic matmult.c

clean:
	rm -f program.o matmult.o
