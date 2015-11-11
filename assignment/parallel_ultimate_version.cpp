#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <thread>
#include <sstream>
#include <string>
#include <ctime>

#define TRAINING_SAMPLE_SIZE 2000

#define TRAINING_INPUT_SIZE 20000
#define HIDDEN_LAYER_SIZE TRAINING_INPUT_SIZE/2

#define ALPHA 0.1

#define PROCESSORS std::thread::hardware_concurrency()

using namespace std;

vector<double> weights(TRAINING_INPUT_SIZE, 0);
vector<double> inputs(TRAINING_INPUT_SIZE, 0);

vector<double> output_weights(HIDDEN_LAYER_SIZE, 0);
vector<double> hidden_nodes(HIDDEN_LAYER_SIZE, 0);
vector<double> hidden_gradients(HIDDEN_LAYER_SIZE, 0);
vector<vector<double>> all_hidden_weights(HIDDEN_LAYER_SIZE, weights);

double gradient = 0;
double wth = 0; //Weights times hidden nodes

vector<double> generate_random_array(int start, int end, int bottom_limit, int upper_limit){
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> distribution(bottom_limit, upper_limit);
	vector<double> random_numbers(end, 0);
	for(int i = start; i < end; i++){
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
	return generate_random_array(0, TRAINING_INPUT_SIZE, -1, 1);
}

void generate_weights_nodes(int hidden_start = 0, int hidden_end = HIDDEN_LAYER_SIZE){
	for (int i = hidden_start; i < hidden_end; i++) {
		all_hidden_weights[i] = generate_random_array(0, TRAINING_INPUT_SIZE, 0, 1);
	}
}

void generate_output_weights(){
	output_weights = generate_random_array(0, HIDDEN_LAYER_SIZE, 0, 1);
}

void calculate_nodes(int hidden_start, int hidden_end){
	for (int i = hidden_start; i < hidden_end; i++) {
		for (int j = 0; j < TRAINING_INPUT_SIZE; j++){
			hidden_nodes[i] += (all_hidden_weights[i][j] * inputs[j]);
		}
		double wtx = hidden_nodes[i];  //Weights times the input
		hidden_nodes[i] = 1 /(1 + exp(-wtx)); //Check the use of 1/1+|x| instead of this, depending on the speed
		// cout << hidden_nodes[i] << endl;
	}
}

void multiply_weights_nodes(int hidden_start, int hidden_end){
	if(hidden_end > HIDDEN_LAYER_SIZE)
		hidden_end = HIDDEN_LAYER_SIZE;
	
	for(int i = hidden_start; i < hidden_end; i++){
		// cout << "i: " << i << " output_weights[i]: "  << output_weights[i] << " hidden_nodes[i]" << hidden_nodes[i] << endl;
		wth += (output_weights[i] * hidden_nodes[i]);
	}
	// cout <<"wth: " << wth << endl;
}

void initialize(){
	generate_weights_nodes();
	generate_output_weights();
	inputs = generate_training_inputs();
}


void update_network(double guess, double classification, 
	int hidden_start, int hidden_end){
	double error = classification - guess;
	double output_gradient = guess * (1 - guess) * error;
	
	for(int i = hidden_start; i < hidden_end; i++){
		double node = hidden_nodes[i];
		hidden_gradients[i] = node * (1 - node) * output_weights[i] * output_gradient;
		output_weights[i] += (ALPHA * output_gradient * node);
		for (int j = 0; j < TRAINING_INPUT_SIZE; j++){
			all_hidden_weights[i][j] += (ALPHA * hidden_gradients[i] * inputs[j]);
		}
	}
}

void parallelize(int classification){
	vector<thread> my_threads;
	const int limit = HIDDEN_LAYER_SIZE/PROCESSORS;
	// cout << "limit: " << limit << endl;
	// cout << "HIDDEN_LAYER_SIZE:" << HIDDEN_LAYER_SIZE << " PROCESSORS: " <<PROCESSORS << endl;
	for(unsigned int i = 1; i < PROCESSORS; i++){
		int hidden_start = (i*limit);
		int hidden_end = limit*(i+1);
		// cout << "hidden_start: " << hidden_start << " hidden_end: " << hidden_end << endl;
		my_threads.push_back(thread(calculate_nodes, hidden_start, hidden_end));
	}
	calculate_nodes(0, limit);
	for (auto i = my_threads.begin(); i != my_threads.end(); ++i){
		i->join();
	}

	my_threads.clear();
	wth = 0;
	for(unsigned int i = 1; i < PROCESSORS; i++){
		int hidden_start = (i*limit);
		int hidden_end = limit*(i+1);
		my_threads.push_back(thread(multiply_weights_nodes, hidden_start, hidden_end));
	}
	multiply_weights_nodes(0, limit);
	for (auto i = my_threads.begin(); i != my_threads.end(); ++i){
		i->join();
	}
	// cout <<"wth: " << wth << endl;

	my_threads.clear();
	double guess = 1 /(1 + exp(-wth));
	// cout << "guess: " << guess << endl << endl;
	for(unsigned int i = 1; i < PROCESSORS; i++){
		int hidden_start = (i*limit);
		int hidden_end = limit*(i+1);
		my_threads.push_back(thread(update_network, guess, classification, hidden_start, hidden_end));
	}
	update_network(guess, classification, 0, limit);
	for (auto i = my_threads.begin(); i != my_threads.end(); ++i){
		i->join();
	}
}

int main(){
	clock_t tStart = clock();
	initialize();
	int classification = 0;
	stringstream wibble;

	for(int i = 0; i < TRAINING_SAMPLE_SIZE; i++){
		inputs = generate_training_inputs();
		classification = wibble_classificator();
		// cout << "classification: " << classification << endl;
		parallelize(classification);

	}
	double excecution_time = (double)(clock() - tStart)/CLOCKS_PER_SEC;
	
	cout << "TRAINING_SAMPLE_SIZE " << TRAINING_SAMPLE_SIZE << endl;
	cout << "TRAINING_INPUT_SIZE " << TRAINING_INPUT_SIZE << endl;

	cout << "Execution time:" << excecution_time << "s" << endl;
	return 0;
}
