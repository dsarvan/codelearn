#!/usr/bin/env python
import invoke

@invoke.task
def program(ctx, path=None):
    """build shared library program.so"""
    ctx.run("gcc -O3 -c -Wall -Werror -fpic program.c")
    ctx.run("gcc -shared -o program.so program.o")

@invoke.task
def matmult(ctx, path=None):
    """build shared library matmult.so"""
    ctx.run("mpicc -O3 -c -Wall -Werror -fpic matmult.c")
    ctx.run("mpicc -shared -o matmult.so matmult.o")

@invoke.task
def laplace(ctx, path=None):
    """build shared library laplace.so"""
    ctx.run("gcc -O3 -c -Wall -Werror -lm -fpic laplace.c")
    ctx.run("gcc -shared -o laplace.so laplace.o")

@invoke.task
def compute(ctx, path=None):
    """build shared library compute.so"""
    ctx.run("gcc -O3 -c -Wall -Werror -fpic compute.c")
    ctx.run("gcc -shared -o compute.so compute.o")

@invoke.task
def clean(ctx, path=None):
    """remove built objects"""
    ctx.run("rm -f program.o matmult.o laplace.o compute.o")

@invoke.task(program, matmult, laplace, compute)
def all(ctx, path=None):
    """build all tasks"""
    pass
