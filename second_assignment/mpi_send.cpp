#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <mpi.h>

#define MPI_TAG 0

#define TRAINING_SAMPLE_SIZE 100

#define TRAINING_INPUT_SIZE 100
#define HIDDEN_LAYER_SIZE TRAINING_INPUT_SIZE/2

#define ALPHA 0.1

using namespace std;

vector<double> weights(TRAINING_INPUT_SIZE, 0);
vector<double> inputs(TRAINING_INPUT_SIZE, 0);

vector<double> output_weights(HIDDEN_LAYER_SIZE, 0);
vector<double> hidden_nodes(HIDDEN_LAYER_SIZE, 0);
vector<double> hidden_gradients(HIDDEN_LAYER_SIZE, 0);
vector<vector<double>> all_hidden_weights(HIDDEN_LAYER_SIZE, weights);

double gradient = 0;

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

int wibble_classificator(){
	inputs[0] = 1;
	int array_size = inputs.size();
	int r[] = {0, 1};
	double label = 0;
	for(int i = 0; i < array_size; i++){
		label += (inputs[i] * r[i%2]);
	}
	return label > 1;
}


void generate_training_inputs(){
	inputs = generate_random_array(TRAINING_INPUT_SIZE, -1, 1);
}

void generate_weights_nodes(){
	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		all_hidden_weights[i] = generate_random_array(TRAINING_INPUT_SIZE, 0, 1);
	}
}

void generate_output_weights(){
	output_weights = generate_random_array(HIDDEN_LAYER_SIZE, 0, 1);
}

void calculate_nodes(int weight_start, int weight_end){
	for (int i = weight_start; i < weight_end; i++) {
		hidden_nodes[i] = 0;
		for (int j = 0; j < TRAINING_INPUT_SIZE; j++){
			hidden_nodes[i] += (all_hidden_weights[i][j] * inputs[j]);
		}
		double wtx = hidden_nodes[i];  //Weights times the input
		hidden_nodes[i] = 1 /(1 + exp(-wtx)); 
	}
}
double nodes_times_weights(int nodes_start, int nodes_end){
	double ntw = 0;  //hidden Nodes Times Weights
	for(int i = nodes_start; i < nodes_end; i++){
		ntw += (output_weights[i] * hidden_nodes[i]);
	}
	return ntw;
}

double calculate_guess_label(double wth){
	double guess = 1 /(1 + exp(wth * -1));
	return guess;
}

void initialize(){
	generate_weights_nodes();
	generate_output_weights();
	generate_training_inputs();
}

void update_network(double guess, double classification){
	double error = classification - guess;
	double output_gradient = guess * (1 - guess) * error;
	
	for(int i = 0; i < HIDDEN_LAYER_SIZE; i++){
		double node = hidden_nodes[i];
		hidden_gradients[i] = node * (1 - node) * output_weights[i] * output_gradient;
		output_weights[i] += (ALPHA * output_gradient * node);
		for (int j = 0; j < TRAINING_INPUT_SIZE; j++){
			all_hidden_weights[i][j] += (ALPHA * hidden_gradients[i] * inputs[j]);
		}
	}
}

int main(){
	MPI_Init(NULL, NULL);
	initialize();
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	for(int i = 0; i < TRAINING_SAMPLE_SIZE; i++){
		generate_training_inputs();
		if(world_rank == world_size - 1){
			int classification = wibble_classificator();
			int start = 0;
			int end = 0;
			for(int j = 0; j < world_rank; j++){
				int limit = HIDDEN_LAYER_SIZE/world_size;
				start = (j*limit);
				end = limit*(j+1);
				MPI_Send(&start, 1, MPI_INT, j, MPI_TAG, MPI_COMM_WORLD);
				MPI_Send(&end, 1, MPI_INT, j, MPI_TAG, MPI_COMM_WORLD);
			}
			start = end;
			end = HIDDEN_LAYER_SIZE;
			calculate_nodes(start, end);
			double guess = nodes_times_weights(start, end);
			for(int j = 0; j < world_rank; j++){
				double partial_guess = 0;
				MPI_Recv(&partial_guess, 1, MPI_DOUBLE, j, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				guess += partial_guess;
			}
			guess = calculate_guess_label(guess);
			update_network(guess, classification);
		} else {
			int start = 0;
			int end = 1;
			int source = world_size - 1;
			MPI_Recv(&start, 1, MPI_INT, source, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&end, 1, MPI_INT, source, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			calculate_nodes(start, end);
			double guess = nodes_times_weights(start, end);
			MPI_Send(&guess, 1, MPI_DOUBLE, source, MPI_TAG, MPI_COMM_WORLD);
		}
	}
	cout << "Finalizing rank " << world_rank << endl;
	MPI_Finalize();
	return 0;
}

