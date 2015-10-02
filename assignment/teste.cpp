#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <ctime>

#define TRAINING_INPUT_SIZE 10
#define TRAINING_SAMPLE_SIZE 10

using namespace std;

int wibble_classificator(float training_data[]){
	int array_size = sizeof(training_data);
	int r[] = {0, 1};
	float label = 0;
	for(int i = 0; i < array_size; i++){
		label += (training_data[i] * r[i%2]);
	}
	return label > 1;
}

float* generate_training_inputs(){
	srand(time(NULL));
	float training_input[TRAINING_INPUT_SIZE];
	for(int i = 0; i < TRAINING_INPUT_SIZE; i++){
		float input = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		training_input[i] = input;
	}
	return training_input;
}

void generate_noise(){
	//For each one of the training samples add some noise
}

void generate_input_weights(){
	//Randomly generate the weights, with numbers near to zero.
}

void calculate_nodes(){
	//Generate with the sigmoid formula 
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
	for(int i = 0; i < TRAINING_INPUT_SIZE; i++){
		cout << wibble_classificator(generate_training_inputs()) << endl;
	}
	return 0;
}