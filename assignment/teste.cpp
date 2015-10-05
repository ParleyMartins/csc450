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

vector<vector<float>> weights;
vector<float> hidden_nodes(HIDDEN_LAYER_SIZE, 0);
vector<float> inputs;
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
		weights[i] = generate_random_array(TRAINING_INPUT_SIZE, 0, 1);
	}
}


void generate_nodes(){
	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		for (int j = 0; j < TRAINING_INPUT_SIZE; j++){
			hidden_nodes[j] += weights[i][j] * inputs[j];
		}
	}
}

void generate_nodes_weights(){
	//Also random numbers, near to zero
}

void calculate_guess_label(){
	//Calculate with the sigmoid formula
}

void calculate_delta(){}

void update_weights(){}

//Return the weights

int main(){
	cout << "teste" << endl;
	for(int i = 0; i < TRAINING_INPUT_SIZE; i++){
		inputs = generate_training_inputs();
		classification = wibble_classificator(inputs);
		cout << classification << endl;
	}
	return 0;
}