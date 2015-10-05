#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <ctime>
#include <vector>

#define TRAINING_INPUT_SIZE 10
#define TRAINING_SAMPLE_SIZE 10
#define HIDDEN_LAYER_SIZE TRAINING_INPUT_SIZE/2

using namespace std;

vector<float> output_weights(HIDDEN_LAYER_SIZE, 0);
vector<float> hidden_nodes(HIDDEN_LAYER_SIZE, 0);

vector<float> weights(TRAINING_INPUT_SIZE, 0);
vector<float> inputs(TRAINING_INPUT_SIZE, 0);

int classification = 0;

int wibble_classificator(vector<float> training_data){
	int array_size = training_data.size();
	int r[] = {0, 1};
	float label = 0;
	for(int i = 0; i < array_size; i++){
		label += (training_data[i] * r[i%2]);
	}
	return label > 1;
}

vector<float> generate_random_array(int size, int bottom_limit, int upper_limit){
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> distribution(bottom_limit, upper_limit);
	vector<float> random_numbers(size, 0);
	random_numbers[0] = 1;
	for(int i = 1; i < size; i++){
		float input = distribution(gen);
		random_numbers[i] = input;
	}
	return random_numbers;
}

vector<float> generate_training_inputs(){
	return generate_random_array(TRAINING_INPUT_SIZE, -1, 1);
}

void generate_noise(){
	//For each one of the training samples add some noise
}

void generate_weights_nodes(){
	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		weights = generate_random_array(TRAINING_INPUT_SIZE, 0, 1);
	}
}


void generate_nodes(){
	generate_weights_nodes();
	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		for (int j = 0; j < TRAINING_INPUT_SIZE; j++){
			hidden_nodes[i] += weights[j] * inputs[j];
		}
		float wtx = hidden_nodes[i];  //Weights times the input
		hidden_nodes[i] = 1 /(1 + exp(wtx)); //Check the use of 1/1+|x| instead of this, depending on the speed
	}
}

void generate_output_weights(){
	output_weights = generate_random_array(HIDDEN_LAYER_SIZE, 0, 1);
}

float calculate_guess_label(){
	generate_output_weights();
	float guess = 0;
	for(int i = 0; i < HIDDEN_LAYER_SIZE; i++){
		guess += output_weights[i] * hidden_nodes[i];
	}
	float wth = guess;  //Weights times hidden nodes
	guess = 1 /(1 + exp(wth));
	return guess;
}

void calculate_delta(){}

void update_weights(){}

//Return the weights

int main(){
	inputs = generate_training_inputs();
	classification = wibble_classificator(inputs);
	cout << classification << endl;
	generate_nodes();
	cout << calculate_guess_label() << endl;
	return 0;
}