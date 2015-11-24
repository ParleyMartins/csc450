#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <mpi.h>

#define INPUT_SIZE 20

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

double multiply(int size, double* vector1, double* vector2){
	double partial_result = 0;
	for(int i = 0; i < size; i++){
		partial_result += (vector1[i] * vector2[i]);
	}
	return partial_result;
}

int main(){
	MPI_Init(NULL, NULL);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int limit = INPUT_SIZE/world_size;

	double* vector1 = NULL;
	double* vector2 = NULL;
	double* partial_results = NULL;

	if(world_rank == 0) {
		vector1 = generate_random_array(INPUT_SIZE, 1, 1);
		vector2 = generate_random_array(INPUT_SIZE, 1, 1);
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
		cout << "Guess: " << result << endl;
	}		
	cout << "Finalizing rank " << world_rank << endl;
	MPI_Finalize();
	return 0;
}

