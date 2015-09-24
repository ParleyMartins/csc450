enum activationFuncs {THRESHOLD = 1, SIGMOID, HYPERBOLIC_TANGENT};

class Perceptron {
private:
	std::vector<float> inputVector; //a vector holding the perceptron's inputs
	std::vector<float> weightsVector; //corresponding inputs weights
	int activationFunction

public:
	Perceptron(int inputNumber, int function);
	void inputAt(int inputPos, float inputValue);
	float calculateNet(); //The activation function type
	void adjustWeights(float teachingStep, float output, float target);
	float recall(float red, float green, float blue);
};