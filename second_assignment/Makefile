EXECS=prog
MPICC=mpic++

all: isend

send: mpi_send.cpp
	${MPICC} -o prog -O2 -std=c++11 -pedantic -W -Wextra -Werror -Wall mpi_send.cpp

scatter: mpi_bsend.cpp
	${MPICC} -o prog -O2 -std=c++11 -pedantic -W -Wextra -Werror -Wall mpi_bsend.cpp

isend: mpi_isend.cpp
	${MPICC} -o prog -O2 -std=c++11 -pedantic -W -Wextra -Werror -Wall mpi_isend.cpp

run: 
	mpirun -np 4 ${EXECS}

clean:
	rm ${EXECS}
