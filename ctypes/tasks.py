#!/usr/bin/env python
import invoke


@invoke.task
def scriptrun(c):
	print("script run!")

@invoke.task
def program(c, path=None):
	invoke.run("gcc -O3 -c -Wall -Werror -fpic program.c")
	invoke.run("gcc -shared -o program.so program.o")

@invoke.task
def matmult(c, path=None):
	invoke.run("mpicc -O3 -c -Wall -Werror -fpic matmult.c")
	invoke.run("mpicc -shared -o matmult.so matmult.o")

@invoke.task
def laplace(c, path=None):
	invoke.run("gcc -O3 -c -Wall -Werror -lm -fpic laplace.c")
	invoke.run("gcc -shared -o laplace.so laplace.o")

@invoke.task
def compute(c, path=None):
	invoke.run("gcc -O3 -c -Wall -Werror -fpic compute.c")
	invoke.run("gcc -shared -o compute.so compute.o")

@invoke.task
def clean(c):
	invoke.run("rm -f program.o matmult.o laplace.o compute.o")

@invoke.task(program, matmult, laplace, compute)
def all(c):
	pass
