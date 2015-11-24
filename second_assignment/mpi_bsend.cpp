#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <mpi.h>

#define MPI_TAG 0

#define TRAINING_SAMPLE_SIZE 1

#define TRAINING_INPUT_SIZE 20


using namespace std;


double* hidden_weights;
vector<double> hidden_nodes;

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


void calculate_nodes(int input_size, double* weights, double* inputs){
	for (unsigned int i = 0; i < hidden_nodes.size(); i++) {
		hidden_nodes[i] = 0;
		for (int j = 0; j < input_size; j++){
			hidden_nodes[i] += (weights[j] * inputs[j]);
		}

		double wtx = hidden_nodes[i];  //Weights times the input
		hidden_nodes[i] = 1 /(1 + exp(-wtx)); 
	}
}

double nodes_times_weights(double* hidden_weights){
	double ntw = 0;  //hidden Nodes Times Weights
	for(unsigned int i = 0; i < hidden_nodes.size(); i++){
		ntw += (hidden_weights[i] * hidden_nodes[i]);
	}
	return ntw;
}

double calculate_guess_label(double wth){
	double guess = 1 /(1 + exp(wth * -1));
	return guess;
}

int main(){
	MPI_Init(NULL, NULL);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int limit = TRAINING_INPUT_SIZE/world_size;
	for(int i = 0; i < TRAINING_SAMPLE_SIZE; i++){
		double* weights = NULL;
		double* inputs = NULL;
		double* guesses = NULL;

		if(world_rank == 0) {
			inputs = generate_random_array(TRAINING_INPUT_SIZE, -1, 1);
			weights = generate_random_array(TRAINING_INPUT_SIZE, 0, 1);
			guesses = (double *) malloc(sizeof(double) * world_size);
		}

		double* partial_inputs = (double *) malloc(sizeof(double) * limit);
		double* partial_weights = (double *) malloc(sizeof(double) * limit);
		MPI_Scatter(inputs, limit, MPI_DOUBLE, partial_inputs, limit, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(weights, limit, MPI_DOUBLE, partial_weights, limit, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		int hidden_limit = limit/2;
		hidden_nodes.assign(hidden_limit, 0);
		calculate_nodes(limit, partial_weights, partial_inputs);

		hidden_weights = (double *) malloc(sizeof(double)*hidden_limit);
		hidden_weights = generate_random_array(hidden_limit, 0, 1);

		double partial_guess = nodes_times_weights(hidden_weights);
		MPI_Gather(&partial_guess, 1, MPI_DOUBLE, guesses, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if(world_rank == 0){
			double guess = 0;
			for(int i = 0; i < world_size; i++){
				guess += guesses[i];
			}
			guess = calculate_guess_label(guess);
			cout << "Guess: " << guess << endl;
		}		
	}
	cout << "Finalizing rank " << world_rank << endl;
	MPI_Finalize();
	return 0;
}

