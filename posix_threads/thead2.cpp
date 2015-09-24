#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <pthread.h>
#include <unistd.h>

using namespace std;

int myglobal = 0;
pthread_mutex_t mymutex = PTHREAD_MUTEX_INITIALIZER;

void *thread_function(void *arg){
	int j = 0;
	for(int i = 0; i < 5; i++){
		pthread_mutex_lock(&mymutex);
		j = myglobal;
		j = j + 1;
		myglobal++;
		cout << ".";
		fflush(stdout);
		sleep(1);
		myglobal = j;
		// pthread_mutex_unlock(&mymutex);
	}
	return NULL;
}

int main(void){
	pthread_t mythread;

	if(pthread_create(&mythread, NULL, thread_function, NULL)){
		cout << "Error catching thread." << endl;
		abort();
	}

	for (int i = 0; i < 5; i++){
		// pthread_mutex_lock(&mymutex);
		myglobal = myglobal + 1;
		cout << "o";
		// pthread_mutex_unlock(&mymutex);
		fflush(stdout);
		sleep(1);
	}

	if(pthread_join(mythread, NULL)){
		cout << "Error joining thread" << endl;
		abort();
	}

	cout << "My global equals " << myglobal << endl;
	exit(0);
}