#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <mpi.h>

#define TRAINING_INPUT_SIZE 20


using namespace std;

vector<double> vector1;
vector<double> vector2;

vector<double> generate_random_array(int size, int bottom_limit, int upper_limit){
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> distribution(bottom_limit, upper_limit);
	vector<double> random_numbers(size, 0);
	for(int i = 0; i < size; i++){
		double input = distribution(gen);
		random_numbers[i] = input;
	}
	return random_numbers;
}

double multiply(){
	double partial_result = 0;
	for(unsigned int i = 0; i < vector1.size(); i++){
		partial_result += (vector1[i] * vector2[i]);
	}
	return partial_result;
}

void initialize(const int size){
	vector1.assign(size, 0.5);// = generate_random_array(size, -1, 1);
	vector2.assign(size, 1);// = generate_random_array(size, 0, 1);
}

int main(){
	MPI_Init(NULL, NULL);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int limit = TRAINING_INPUT_SIZE/world_size;
	int MPI_TAG = 0;
	//for(int i = 0; i < TRAINING_SAMPLE_SIZE; i++){
		initialize(limit);
		if(world_rank == world_size - 1){
			double guess = multiply();
			for(int j = 0; j < world_rank; j++){
				double partial_guess = 0;
				MPI_Recv(&partial_guess, 1, MPI_DOUBLE, j, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				cout << "Partial Guess: " << partial_guess << endl;
				guess += partial_guess;
			}
			cout << "Guess: " << guess << endl;
		} else {
			double guess = multiply();
			MPI_Request send_request;
			MPI_Isend(&guess, 1, MPI_DOUBLE, world_size - 1, MPI_TAG, MPI_COMM_WORLD, &send_request);
		}
//	}
	cout << "Finalizing rank " << world_rank << endl;
	MPI_Finalize();
	return 0;
}

