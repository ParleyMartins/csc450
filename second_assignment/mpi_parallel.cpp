#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <ctime>
#include <mpi.h>

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


vector<double> generate_training_inputs(){
	return generate_random_array(TRAINING_INPUT_SIZE, -1, 1);
}

void generate_weights_nodes(){
	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		all_hidden_weights[i] = generate_random_array(TRAINING_INPUT_SIZE, 0, 1);
	}
}

void generate_output_weights(){
	output_weights = generate_random_array(HIDDEN_LAYER_SIZE, 0, 1);
}

void calculate_nodes(int hidden_start, int hidden_end, int input_start, int input_end){
	for (int i = hidden_start; i < hidden_end; i++) {
		for (int j = input_start; j < input_end; j++){
			hidden_nodes[i] += (all_hidden_weights[i][j] * inputs[j]);
		}
		double wtx = hidden_nodes[i];  //Weights times the input
		hidden_nodes[i] = 1 /(1 + exp(-wtx)); //Check the use of 1/1+|x| instead of this, depending on the speed
	}
}

double calculate_guess_label(){
	double guess = 0;
	for(int i = 0; i < HIDDEN_LAYER_SIZE; i++){
		// cout <<"i: " << i << " output_weights[i]: "  << output_weights[i] << " hidden_nodes[i]" << hidden_nodes[i] << endl;
		guess += (output_weights[i] * hidden_nodes[i]);
	}
	double wth = guess;  //Weights times hidden nodes
	// cout <<"wth: " << wth << endl;
	guess = 1 /(1 + exp(-wth));
	return guess;
}

void initialize(){
	generate_weights_nodes();
	generate_output_weights();
	inputs = generate_training_inputs();
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
	clock_t tStart = clock();
	initialize();
	int classification = 0;
	double guess = 1;

	for(int i = 0; i < TRAINING_SAMPLE_SIZE; i++){
		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		if(world_size > 1){
			int world_rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
			if(world_rank != 0){
			}
		}
		inputs = generate_training_inputs();
		classification = wibble_classificator();
		// cout << "classification: " << classification << endl;
		calculate_nodes(0, HIDDEN_LAYER_SIZE, 0, TRAINING_INPUT_SIZE);
		guess = calculate_guess_label();
		// cout << "guess: " << guess << endl << endl;
		update_network(guess, classification);
	}

   
	double excecution_time = (double)(clock() - tStart)/CLOCKS_PER_SEC;
	cout << "TRAINING_SAMPLE_SIZE " << TRAINING_SAMPLE_SIZE << endl;
	cout << "TRAINING_INPUT_SIZE " << TRAINING_INPUT_SIZE << endl;

	cout << "Execution time:" << excecution_time << "s" << endl;
	MPI_Finalize();
	return 0;
}

