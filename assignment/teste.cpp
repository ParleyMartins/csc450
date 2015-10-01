#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

int give_label(float training_data[]){
	int array_size = sizeof(training_data);
	int r[] = {0, 1};
	float label = 0;
	cout << "array size " << array_size << endl;
	for(int i = 0; i < array_size; i++){
		label += (training_data[i] * r[i%2]);
		cout << "td: " << training_data[i] << endl;
		cout << "i%2: " << i%2 << " r[i%2]: " << r[i%2]  << endl;
		cout << "label: " << label << endl;
	}
	return round(label);
}

int main(){
	float test[] = {0.7,
-0.8,
0.9,
0.631,
-0.842,
0.861,
0.976,
-0.784};
	cout << give_label(test) << endl;
	return 0;
}