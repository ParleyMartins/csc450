EXECS=prog
MPICC=mpic++

all: ${EXECS}

prog: mpi_parallel.cpp
	${MPICC} -o prog -O2 -std=c++11 -pedantic -W -Wall mpi_parallel.cpp

run: 
	mpirun -np 4 ${EXECS}

clean:
	rm ${EXECS}
