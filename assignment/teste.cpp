#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <ctime>
#include <vector>

#define TRAINING_INPUT_SIZE 1000
#define HIDDEN_LAYER_SIZE TRAINING_INPUT_SIZE/2
#define ALPHA 0.01

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
	random_numbers[0] = 1;
	for(int i = 1; i < size; i++){
		double input = distribution(gen);
		random_numbers[i] = input;
	}
	return random_numbers;
}

int wibble_classificator(vector<double> training_data){
	int array_size = training_data.size();
	int r[] = {0, 1};
	double label = 0;
	for(int i = 0; i < array_size; i++){
		label += (training_data[i] * r[i%2]);
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

void generate_nodes(){
	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		for (int j = 0; j < TRAINING_INPUT_SIZE; j++){
			hidden_nodes[i] += all_hidden_weights[i][j] * inputs[j];
		}
		double wtx = hidden_nodes[i];  //Weights times the input
		hidden_nodes[i] = 1 /(1 + exp(wtx)); //Check the use of 1/1+|x| instead of this, depending on the speed
	}
	
}

void generate_output_weights(){
	output_weights = generate_random_array(HIDDEN_LAYER_SIZE, 0, 1);
}

double calculate_guess_label(){
	double guess = 0;
	for(int i = 0; i < HIDDEN_LAYER_SIZE; i++){
		guess += output_weights[i] * hidden_nodes[i];
	}
	double wth = guess;  //Weights times hidden nodes
	guess = 1 /(1 + exp(wth));
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
	}

	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		for (int j = 0; j < TRAINING_INPUT_SIZE; j++){
			all_hidden_weights[i][j] += (ALPHA * hidden_gradients[i] * inputs[j]);
		}
	}

	
}

int main(){
	initialize();
	int classification = 0;
	double guess = 1;
	string teste = " ";
	for(int i = 0; i < 50; i++){
		inputs = generate_training_inputs();
		classification = wibble_classificator(inputs);
		int j = 0;
		while(fabs(classification - guess) > 0.000001){
			cout << "iteration: "<< i << " " << j << endl;
			cout << classification << endl;
			generate_nodes();
			double second_guess = calculate_guess_label();
			cout << second_guess << endl;
			if(fabs(second_guess - guess) < 0.000001)
				break;
			guess = second_guess;
			update_network(guess, classification);
			j++;
			// cin >> teste;
		}
	}
	for (int i = 0; i < TRAINING_INPUT_SIZE; ++i)
	{
		cout << "weights[0][" << i << "]: " << all_hidden_weights[0][i] << endl;
	}
	return 0;
}

// #define TRAINING_SAMPLE_SIZE 10

// void generate_noise() //For each one of the training samples add some noise
// void calculate_delta(){}
// void update_weights(){}
