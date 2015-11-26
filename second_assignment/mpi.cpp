/*
 * This program consists in implementing a vector multiplication.
 * Both vectors are generated randomly.
 */

#include <iostream>
#include <cstdlib>
#include <random>
#include <mpi.h>
#include <sstream>

#define V1_TAG 10
#define V2_TAG 20
#define R_TAG 30

#define SCATTER 2
#define ISEND 3

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
double multiply(int size, double* vector1, double* vector2, int start = 0){
	double partial_result = 0;
	for(int i = start; i < size; i++){
		partial_result += (vector1[i] * vector2[i]);
	}
	return partial_result;
}

void scatter_gather(unsigned const int INPUT_SIZE, const int world_size, int world_rank){	
	int limit = INPUT_SIZE/world_size;

	//Pointers used by the root process to scatter the data
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

	//0 is the root process
	MPI_Scatter(vector1, limit, MPI_DOUBLE, partial_v1, limit, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(vector2, limit, MPI_DOUBLE, partial_v2, limit, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double partial_result = multiply(limit, partial_v1, partial_v2);
	// 1 is the amount of data being passed and 0 is the root process in this call
	MPI_Gather(&partial_result, 1, MPI_DOUBLE, partial_results, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(world_rank == 0){
		double result = 0;
		for(int i = 0; i < world_size; i++){
			result += partial_results[i];
		}
		cout << "Result: " << result << endl;
	}		
}

void send(unsigned const int INPUT_SIZE, const int world_size, int world_rank){	
	int limit = INPUT_SIZE/world_size;

	//Pointers used by the root process to scatter the data
	double* vector1 = NULL;
	double* vector2 = NULL;

	
	if(world_rank == world_size - 1) {
		//The first process initializes the values of all the arrays.
		vector1 = generate_random_array(INPUT_SIZE, 1, 1);
		vector2 = generate_random_array(INPUT_SIZE, 1, 1);

		for(int i = 0; i < world_rank; i++){
			MPI_Send(vector1 + (i*limit), limit, MPI_DOUBLE, i, V1_TAG, MPI_COMM_WORLD);
			MPI_Send(vector2 + (i*limit), limit, MPI_DOUBLE, i, V2_TAG, MPI_COMM_WORLD);
		}
		double result = multiply(INPUT_SIZE, vector1, vector2, world_rank*limit);
		
		for(int i = 0; i < world_rank; i++){
			double partial_result = 0;
			MPI_Recv(&partial_result, 1, MPI_DOUBLE, i, R_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			result += partial_result;
		}
		cout << "Result: " << result << endl;
	} else {
		double* partial_v1 = (double *) malloc(sizeof(double) * limit);
		double* partial_v2 = (double *) malloc(sizeof(double) * limit);

		MPI_Recv(partial_v1, limit, MPI_DOUBLE, world_size - 1, V1_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(partial_v2, limit, MPI_DOUBLE, world_size - 1, V2_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		double partial_result = multiply(limit, partial_v1, partial_v2);
		MPI_Send(&partial_result, 1, MPI_DOUBLE, world_size - 1, R_TAG, MPI_COMM_WORLD);
	}
}


void isend(unsigned const int INPUT_SIZE, const int world_size, int world_rank){	
	int limit = INPUT_SIZE/world_size;

	//Pointers used by the root process to scatter the data
	double* vector1 = NULL;
	double* vector2 = NULL;

	
	if(world_rank == world_size - 1) {
		//The first process initializes the values of all the arrays.
		vector1 = generate_random_array(INPUT_SIZE, 1, 1);
		vector2 = generate_random_array(INPUT_SIZE, 1, 1);

		for(int i = 0; i < world_rank; i++){
			MPI_Request request;
			MPI_Isend(vector1 + (i*limit), limit, MPI_DOUBLE, i, V1_TAG, MPI_COMM_WORLD, &request);
			MPI_Isend(vector2 + (i*limit), limit, MPI_DOUBLE, i, V2_TAG, MPI_COMM_WORLD, &request);
		}
		double result = multiply(INPUT_SIZE, vector1, vector2, world_rank*limit);
		for(int i = 0; i < world_rank; i++){
			double partial_result = 0;
			MPI_Recv(&partial_result, 1, MPI_DOUBLE, i, R_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			result += partial_result;
		}
		cout << "Result: " << result << endl;
	} else {

		double* partial_v1 = (double *) malloc(sizeof(double) * limit);
		double* partial_v2 = (double *) malloc(sizeof(double) * limit);

		MPI_Recv(partial_v1, limit, MPI_DOUBLE, world_size - 1, V1_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(partial_v2, limit, MPI_DOUBLE, world_size - 1, V2_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		double partial_result = multiply(limit, partial_v1, partial_v2);
		MPI_Request request;
		MPI_Isend(&partial_result, 1, MPI_DOUBLE, world_size - 1, R_TAG, MPI_COMM_WORLD, &request);
	}
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
	
	unsigned int input_size = 200000;
	int choice = 0;
	//check if a size was given
	if(argc >= 3){
		//Using sstream to convert the value
		stringstream ss;
		ss << argv[1] << ' ' << argv[2];
		if(!(ss >> choice))
			cerr << "1st argument should be 2 for scatter, 3 for isend or any other integer for default send" << endl;
		if(!(ss >> input_size))
			cerr << "2nd argument should be a positive integer, because that is going to be the size of your vectors" << endl;
	}
	
	switch(choice){
		case SCATTER:
			scatter_gather(input_size, world_size, world_rank);
			break;
		case ISEND:
			isend(input_size, world_size, world_rank);
			break;
		default:
			cout << "(No arguments or default send choosen)" << endl;
			send(input_size, world_size, world_rank);
	}
	MPI_Finalize();
	return 0;
}
