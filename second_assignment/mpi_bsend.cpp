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
 * Multiply the partial vectors. To avoid global variables, receives both vectors and the size.
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
		istringstream ss(argv[1]);
		if(!(ss >> INPUT_SIZE))
			cerr << "1st argument should be a positive integer" << endl;
	}
	int limit = INPUT_SIZE/world_size;

	double* vector1 = NULL;
	double* vector2 = NULL;
	double* partial_results = NULL;

	if(world_rank == 0) {
		//The first process initializes the values of all the arrays.
		vector1 = generate_random_array(INPUT_SIZE, 1, 1);
		vector2 = generate_random_array(INPUT_SIZE, 1, 1);

		//This is here to avoid one more comparison
		partial_results = (double *) malloc(sizeof(double) * world_size);
	}

	double* partial_v1 = (double *) malloc(sizeof(double) * limit);
	double* partial_v2 = (double *) malloc(sizeof(double) * limit);
	MPI_Scatter(vector1, limit, MPI_DOUBLE, partial_v1, limit, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(vector2, limit, MPI_DOUBLE, partial_v2, limit, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double partial_result = multiply(limit, partial_v1, partial_v2);
	MPI_Gather(&partial_result, 1, MPI_DOUBLE, partial_results, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(world_rank == 0){
		double result = 0;
		for(int i = 0; i < world_size; i++){
			result += partial_results[i];
		}
		cout << "Result: " << result << endl;
	}		
	MPI_Finalize();
	return 0;
}

