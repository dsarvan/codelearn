all: program.so matmult.so laplace.so compute.so

program.so: program.o
	gcc -shared -o program.so program.o

program.o: program.c
	gcc -O3 -c -Wall -Werror -fpic program.c

matmult.so: matmult.o
	mpicc -shared -o matmult.so matmult.o

matmult.o: matmult.c
	mpicc -O3 -c -Wall -Werror -fpic matmult.c

laplace.so: laplace.o
	gcc -shared -o laplace.so laplace.o

laplace.o: laplace.c
	gcc -O3 -c -Wall -Werror -lm -fpic laplace.c

compute.so: compute.o
	gcc -shared -o compute.so compute.o

compute.o: compute.c
	gcc -O3 -c -Wall -Werror -fpic compute.c

clean:
	rm -f program.o matmult.o laplace.o compute.o
