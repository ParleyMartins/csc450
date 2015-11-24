EXECS=prog
MPICC=mpic++

all: scatter

send: mpi_send.cpp
	${MPICC} -o prog -O2 -std=c++11 -pedantic -W -Wextra -Werror -Wall mpi_send.cpp

scatter: mpi_send.cpp
	${MPICC} -o prog -O2 -std=c++11 -pedantic -W -Wextra -Werror -Wall mpi_send.cpp

isend: mpi_send.cpp
	${MPICC} -o prog -O2 -std=c++11 -pedantic -W -Wextra -Werror -Wall mpi_send.cpp

run: 
	mpirun -np 4 ${EXECS}

clean:
	rm ${EXECS}
