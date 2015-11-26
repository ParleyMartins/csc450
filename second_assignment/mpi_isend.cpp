/*
 * This program consists in implementing a vector multiplication.
 * Both vectors are generated randomly.
 */

#include <iostream>
#include <cstdlib>
#include <random>
#include <mpi.h>
#include <sstream>

using namespace std;
/*
 * Generates random array (using c++11 library) within given limits
  */
double* generate_random_array(int size, int bottom_limit, int upper_limit){
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> distribution(bottom_limit, upper_limit);
	double* random_numbers = (double*) malloc(sizeof(double) * size);
	for(int i = 0; i < size; i++){
		double input = distribution(gen);
		random_numbers[i] = input;
	}
	return random_numbers;
}

/*
 * Multiplies the partial vectors. To avoid global variables, receives both vectors and the size.
 */
double multiply(int size, double* vector1, double* vector2){
	double partial_result = 0;
	for(int i = 0; i < size; i++){
		partial_result += (vector1[i] * vector2[i]);
	}
	return partial_result;
}

/*
 * This function receives the size of the vectors as parameter. If none provided,
 * it will use 200000 as default 
 */
int main(int argc, char* argv[]){
	// Standard MPI initializers	
	MPI_Init(&argc, &argv);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	unsigned int INPUT_SIZE = 200000;
	//check if a size was given
	if(argc >= 2){
		//Using sstream to convert the value
		istringstream ss(argv[1]);
		if(!(ss >> INPUT_SIZE))
			cerr << "1st argument should be a positive integer" << endl;
	}
	int limit = INPUT_SIZE/world_size;

	//Pointers used by the root process to scatter the data
	double* vector1 = NULL;
	double* vector2 = NULL;
	double* partial_results = NULL;
	
	if(world_rank == world_size - 1){
		for(int j = 0; j < world_rank; j++){
			MPI_Isend(&result, 1, MPI_DOUBLE, world_size - 1, MPI_TAG, MPI_COMM_WORLD, &send_request);
		}
		cout << "Guess: " << result << endl;
	} else {
		double result = multiply();
		MPI_Request send_request;
		MPI_Isend(&result, 1, MPI_DOUBLE, world_size - 1, MPI_TAG, MPI_COMM_WORLD, &send_request);
	}
	cout << "Finalizing rank " << world_rank << endl;
	MPI_Finalize();
	return 0;
}

