EXECS=prog
MPICC=mpic++

all: send

send: mpi.cpp
	${MPICC} -o prog -O2 -std=c++11 -pedantic -W -Wextra -Werror -Wall mpi.cpp

run: 
	mpirun -np 4 ${EXECS}

clean:
	rm ${EXECS}
