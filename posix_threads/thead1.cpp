#include <cstdlib>
#include <iostream>

#include <pthread.h>
#include <unistd.h>

using namespace std;

void  *thread_function(void *arg){
	for(int i = 0; i < 20; i++){
		cout << "Thread says hi!" << endl;
		sleep(1);
	}

	return NULL;
}

int main(void){
	pthread_t mythread;

	if(pthread_create(&mythread, NULL, thread_function, NULL)){
		cout << "Error catching thread." << endl;
		abort();
	}

	if(pthread_join(mythread, NULL)){
		cout << "Error joining thread." << endl;
		abort();
	}

	exit(0);
}