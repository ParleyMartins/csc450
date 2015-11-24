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

vector<double> weights;
vector<double> inputs;

vector<double> hidden_weights;
vector<double> hidden_nodes;

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

void calculate_nodes(){
	for (unsigned int i = 0; i < hidden_nodes.size(); i++) {
		hidden_nodes[i] = 0;
		for (unsigned int j = 0; j < inputs.size(); j++){
			hidden_nodes[i] += (weights[j] * inputs[j]);
		}

		double wtx = hidden_nodes[i];  //Weights times the input
		hidden_nodes[i] = 1 /(1 + exp(-wtx)); 
	}
}
double nodes_times_weights(){
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

void initialize(const int size){
	inputs = generate_random_array(size, -1, 1);
	weights = generate_random_array(size, 0, 1);
	int hidden_size = size/2;
	hidden_weights = generate_random_array(hidden_size, 0, 1); 
	hidden_nodes.assign(hidden_size, 0);
}

int main(){
	MPI_Init(NULL, NULL);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int limit = TRAINING_INPUT_SIZE/world_size;
	for(int i = 0; i < TRAINING_SAMPLE_SIZE; i++){
		if(world_rank == 0) {
			initialize(TRAINING_INPUT_SIZE);	
		}
		MPI_Scatter(inputs, limit, MPI_DOUBLE, inputs, limit, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
	}
	cout << "Finalizing rank " << world_rank << endl;
	MPI_Finalize();
	return 0;
}

