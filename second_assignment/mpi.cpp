/*
 * This program consists in implementing a vector multiplication.
 * Both vectors are generated randomly.
 */

#include <iostream>
#include <cstdlib>
#include <random>
#include <mpi.h>
#include <sstream>

/*
 * Used three different tags to ensure that the receive functions
 * would get only what it was suppose to i wach turn.
 * V1 for first vector
 * V2 for second vector
 * R for result
 */
#define V1_TAG 10
#define V2_TAG 20
#define R_TAG 30

/*
 * Define these constants to avoid using meaningless numbers in the code
 */
#define SCATTER 2
#define ISEND 3

using namespace std;
/*
 * Generates random int array (using c++11 library) within given limits
 * To ensure correctnes of the program, it was passed 1 as both limits, generating two vectors filled with 1.
 * The multiplication should generate a correct value, which happened successfully
  */
int* generate_random_array(int size, int bottom_limit, int upper_limit){
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> distribution(bottom_limit, upper_limit);
	int* random_numbers = (int*) malloc(sizeof(int) * size);
	for(int i = 0; i < size; i++){
		int input = distribution(gen);
		random_numbers[i] = input;
	}
	return random_numbers;
}

/*
 * Multiplies the partial vectors. To avoid global variables, receives both vectors and the size.
 * The start point is for the root process, that has all the data in both vectors. Since the division is not always exact,
 * the root start point is the last value sent and it goes all the way to the end to ensure correctness.
 */
int multiply(int size, int* vector1, int* vector2, int start = 0){
	int partial_result = 0;
	for(int i = start; i < size; i++){
		partial_result += (vector1[i] * vector2[i]);
	}
	return partial_result;
}

void scatter_gather(unsigned const int INPUT_SIZE, const int world_size, int world_rank){	
	int limit = INPUT_SIZE/world_size;

	//Pointers used by the root process to scatter the data
	int* vector1 = NULL;
	int* vector2 = NULL;
	int* partial_results = NULL;

	if(world_rank == 0) {
		//The first process initializes the values of all the arrays.
		vector1 = generate_random_array(INPUT_SIZE, 1, 1);
		vector2 = generate_random_array(INPUT_SIZE, 1, 1);

		//This is here to avoid one more comparison
		partial_results = (int *) malloc(sizeof(int) * world_size);
	}

	/*
 	* Because the size of the input may not be divisible by the number of processes, it's used the Scatterv function
 	* that sends different amounts of data to each process. The same thought of the default send is applied and the last
 	* process receives the bigger data.
 	*/
	int* sendcounts = (int *) malloc(sizeof(int) * world_size);
	int* displacements = (int *) malloc(sizeof(int) * world_size);
	
	for(int i = 0; i < world_size; i++){
		sendcounts[i] = limit;
		displacements[i] = i*limit;
	}
	sendcounts[world_size - 1] = INPUT_SIZE - displacements[world_size - 1];
	
	int* partial_v1 = (int *) malloc(sizeof(int) * sendcounts[world_rank]);
	int* partial_v2 = (int *) malloc(sizeof(int) * sendcounts[world_rank]);
	/*
	 * The Scatterv implementation is, according to the documentation, is equivalent to the send function in
	 * this program. Because of this, I would prefer to use the standard send/receive instead of the scatterv,
	 * since this last is much more complicated to read.
	 * However, when you are certain that the amount of data sent to each process will be the same, using scatter
	 * may be a good option. As a good point it's easier to read but you will need  more memory to generate
	 * a receive buffer even for the root process;
	 */
	//0 is the root process
	MPI_Scatterv(vector1, sendcounts, displacements, MPI_INT, partial_v1, sendcounts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(vector2, sendcounts, displacements, MPI_INT, partial_v2, sendcounts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

	int partial_result = multiply(sendcounts[world_rank], partial_v1, partial_v2);
	// 1 is the amount of data being passed and 0 is the root process in this call
	MPI_Gather(&partial_result, 1, MPI_INT, partial_results, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if(world_rank == 0){
		int result = 0;
		for(int i = 0; i < world_size; i++){
			result += partial_results[i];
		}
		cout << "Result: " << result << endl;
	}		
}

void send(unsigned const int INPUT_SIZE, const int world_size, int world_rank){	
	int limit = INPUT_SIZE/world_size;

	//Pointers used by the root process to scatter the data
	int* vector1 = NULL;
	int* vector2 = NULL;

	/*
	 * The last process was chosen as root because it's easier to iterate through the last portion of the array
	 * because of uneven division. So the last process handles all the data left.
	 */
	if(world_rank == world_size - 1) {
		//The root process initializes the values of all the arrays.
		vector1 = generate_random_array(INPUT_SIZE, 1, 1);
		vector2 = generate_random_array(INPUT_SIZE, 1, 1);

		for(int i = 0; i < world_rank; i++){
			MPI_Send(vector1 + (i*limit), limit, MPI_INT, i, V1_TAG, MPI_COMM_WORLD);
			MPI_Send(vector2 + (i*limit), limit, MPI_INT, i, V2_TAG, MPI_COMM_WORLD);
		}
		int result = multiply(INPUT_SIZE, vector1, vector2, world_rank*limit);
		
		for(int i = 0; i < world_rank; i++){
			int partial_result = 0;
			MPI_Recv(&partial_result, 1, MPI_INT, i, R_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			result += partial_result;
		}
		cout << "Result: " << result << endl;
	} else {
		int* partial_v1 = (int *) malloc(sizeof(int) * limit);
		int* partial_v2 = (int *) malloc(sizeof(int) * limit);

		MPI_Recv(partial_v1, limit, MPI_INT, world_size - 1, V1_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(partial_v2, limit, MPI_INT, world_size - 1, V2_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		int partial_result = multiply(limit, partial_v1, partial_v2);
		MPI_Send(&partial_result, 1, MPI_INT, world_size - 1, R_TAG, MPI_COMM_WORLD);
	}
}


void isend(unsigned const int INPUT_SIZE, const int world_size, int world_rank){	
	int limit = INPUT_SIZE/world_size;

	//Pointers used by the root process to scatter the data
	int* vector1 = NULL;
	int* vector2 = NULL;

	
	if(world_rank == world_size - 1) {
		//The first process initializes the values of all the arrays.
		vector1 = generate_random_array(INPUT_SIZE, 1, 1);
		vector2 = generate_random_array(INPUT_SIZE, 1, 1);

		for(int i = 0; i < world_rank; i++){
			MPI_Request request;
			MPI_Isend(vector1 + (i*limit), limit, MPI_INT, i, V1_TAG, MPI_COMM_WORLD, &request);
			MPI_Isend(vector2 + (i*limit), limit, MPI_INT, i, V2_TAG, MPI_COMM_WORLD, &request);
		}
		int result = multiply(INPUT_SIZE, vector1, vector2, world_rank*limit);
		for(int i = 0; i < world_rank; i++){
			//Since the receive function uses a pointer, I had to create a new variable here to receive it
			int partial_result = 0;
			MPI_Recv(&partial_result, 1, MPI_INT, i, R_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			result += partial_result;
		}
		cout << "Result: " << result << endl;
	} else {
		//Buffers to receive the partial vector send by last process
		int* partial_v1 = (int *) malloc(sizeof(int) * limit);
		int* partial_v2 = (int *) malloc(sizeof(int) * limit);
		MPI_Recv(partial_v1, limit, MPI_INT, world_size - 1, V1_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(partial_v2, limit, MPI_INT, world_size - 1, V2_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
		//Store the partial result to send to last process sum	
		int partial_result = multiply(limit, partial_v1, partial_v2);
		MPI_Request request;
		MPI_Isend(&partial_result, 1, MPI_INT, world_size - 1, R_TAG, MPI_COMM_WORLD, &request);
	}
}



/*
 * This function receives the send protocol and the size of the vectors. 
 * If none provided, it will use 200000 and send (blocking) as default.
 */
int main(int argc, char* argv[]){
	// Standard MPI initializers
	MPI_Init(&argc, &argv);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	unsigned int input_size = 200000;
	int choice = 0;
	//check if a size was given
	if(argc >= 3){
		//Using sstream to convert the value
		stringstream ss;
		ss << argv[1] << ' ' << argv[2];
		if(!(ss >> choice))
			cerr << "1st argument should be the mpi protocol to be used (1, 2, 3)" << endl;
		if(!(ss >> input_size))
			cerr << "2nd argument should be a positive integer (size of your vectors)" << endl;
	} else if (world_rank == 0){
		cout << "****************" << endl;
		cout << "No arguments found for the program." << endl;
		cout << "Running it with default configurations (block send and 200000 as vector size)" << endl;	
		cout << "To change these settings pass:" << endl;
		cout << "1, 2 or 3 as first param to choose the MPI protocol (send, scatter or isend, respectively);" << endl;
		cout << "A positive integer as second param to be your vector size;" << endl;
		cout << "****************" << endl << endl;
	}
	
	switch(choice){
		case SCATTER:
			scatter_gather(input_size, world_size, world_rank);
			break;
		case ISEND:
			isend(input_size, world_size, world_rank);
			break;
		default:
			send(input_size, world_size, world_rank);
	}
	MPI_Finalize();
	return 0;
}
