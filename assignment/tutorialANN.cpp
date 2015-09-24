#include <iostream>
#include <cstdlib>
#include "perceptron.h"

using namespace std;

int ourInput[] = {
    //RED GREEN BLUE CLASS
    0, 0, 255, CLASS_BLUE,
    0, 0, 192, CLASS_BLUE,
    243, 80, 59, CLASS_RED,
    255, 0, 77, CLASS_RED,
    77, 93, 190, CLASS_BLUE,
    255, 98, 89, CLASS_RED,
    208, 0, 49, CLASS_RED,
    67, 15, 210, CLASS_BLUE,
    82, 117, 174, CLASS_BLUE,
    168, 42, 89, CLASS_RED,
    248, 80, 68, CLASS_RED,
    128, 80, 255, CLASS_BLUE,
    228, 105, 116, CLASS_RED
    };
int main() {
    Perceptron ann(3, SIGMOID); //# inputs and Sigmoid as activation
    float mse = 999; //Mean Square Error (e^2)
    int epochs = 0;

    //The Training of the ANN
    while(fabs(mse - LEAST_MEAN_SQUARE_ERROR) > 0.0001){
        mse = 0;
        float error = 0;
        inputCounter  0;
        //Run through all 13 input patterns, what we call an EPOCH
        for(int j = 0; j < inputPatterns, j++){
            for(int k = 0; k < 3; k++){
                ann.inputAt(k, normalize(ourInput[inputCounter]))
                inputCounter++;
            }
            //Let's get the output of this particular RGB pattern
            output = ann.calculateNet();
            error += fabs(ourInput[inputCounter] - output) //Add the error

            //And adjust the weights according to that error
            ann.adjustWeights(TEACHING_STEP, output, ourInput[inputCounter]);
            inputCounter++;
        }
        mse = error/inputPatterns //Compute the mean square error to this epoch
        cout << "The mean square error of " << epochs << " is " << mse << endl;
        epochs++;
    }
    return 0;
}