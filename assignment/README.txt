C++ 11 (-std=c++11) and warning flags (-W -Wall) were used to compile the files.
Even though the use of warning flags is optional, c++11 must be enable in order to don't get any errors.

The output is amount of samples (TRAINING_SAMPLE_SIZE), the size of the input (TRAINING_INPUT_SIZE) and the execution time for those.


g++ -W -Wall -pedantic -std=c++11 -O2ser -o serial serial.cpp
g++ -W -Wall -pedantic -std=c++11 -O2ser -o parallel parallel.cpp