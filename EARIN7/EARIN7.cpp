// EARIN7.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>

#define p1 2
#define p2 4
double LEARING_RATE =  0.2;


//f : [-10,10] -> R   f=sin(x*sqrt(a)) + cos(x*sqrt(b))   a = 2 + 1, b = 4 +1 
//
double func_f(double x) {
    int a = p1 + 1;
    int b = p2 + 1;

    double result = sin(x * sqrt(a)) + cos(x * sqrt(b));

    return result;
}

//maybe rewrite for proper sigmoid?
double sigmoid(double x) {
	return tanh(x);
}

// 1/cosh2(x)
double sigmoid_derivative(double x) {
	double low = cosh(x);
	low = low * low;

	return 1.0 / low;
}

class Layer;

class Neuron
{
private:
	double neuron_value;
	double bias;
	double neuron_error;
	double neuron_sum;
	std::vector <double> weights;
	std::vector <double> inputs;
	bool input_neuron;

	Layer* prev_layer;

public:
	Neuron(Layer* prev_layer_ptr);
	double GetValue();
	double GetError();

	void PrintWeights();

	void ForwardPass();
	void BackwardPass();
	void SetValue(double val);
	void SetInput(std::vector <double> in_vals);
	void SetError(double error_val);
	void GradientUpdate();
};

class Layer
{
private:
	std::vector <Neuron> neurons;
	bool input_layer;


public:
	Layer(int size,Layer* prev_layer);
	std::vector <double> GetValues();
	int GetSize();

	void PrintWeights();

	void ForwardPass();
	void BackwardPass();
	void SetValue(int index, double value);
	void SetInput(std::vector <double> in_vals);
	void AddError(int index, double val);
	void GradientUpdate();
};

Neuron::Neuron(Layer* prev_layer_ptr) {

	int prev_layer_size;

	if (prev_layer_ptr == NULL) {
		prev_layer_size = 0;
		this->input_neuron = true;
	}
	else {
		prev_layer_size = prev_layer_ptr->GetSize();
		this->input_neuron = false;
	}

	
	//set connections
	for (int i = 0; i < prev_layer_size; i++) {
		this->prev_layer = prev_layer_ptr;
	}
	this->neuron_error = 0;

	//set random bias
	//int rand_int = rand() % 1000;
	//this->bias = (double)(rand_int / 1000.0);
	this->bias = 0.0;

	//set random weights
	for(int i = 0; i < prev_layer_size; i++) {
		int rand_int = rand() % 1000;
		double rand_float = rand_int / 1000.0;
		this->weights.push_back(rand_float);
	}
}

double Neuron::GetValue() {
	return this->neuron_value;
}

double Neuron::GetError() {
	return this->neuron_error;
}

void Neuron::PrintWeights() {
	std::cout << "   ";
	for (double w : this->weights) {
		std::cout << w << " ";
	}
	std::cout << "\n";
}

void Neuron::SetValue(double val) {
	this->neuron_value = val;
}

void Neuron::SetInput(std::vector <double> in_vals) {
	this->inputs = in_vals;
}

void Neuron::SetError(double error_val) {
	this->neuron_error = error_val;
}

void Neuron::GradientUpdate() {
	double grad;

	//for each weigth
	for (int i = 0; i < this->weights.size();i++) {
		grad = this->inputs[i] * sigmoid_derivative(this->neuron_sum) * this->neuron_error * LEARING_RATE;

		//update weight
		this->weights[i] -= grad;
	}
}

void Neuron::ForwardPass() {
	if (input_neuron) return;


	this->inputs = this->prev_layer->GetValues();
	double totalVal = 0;

	//clear previous backwardpass
	this->neuron_error = 0;

	for (int i = 0; i < this->weights.size(); i++) {
		totalVal += this->weights[i] * inputs[i];
	}

	totalVal += this->bias;
	this->neuron_sum = totalVal;
	this->neuron_value = sigmoid(totalVal);

}

void Neuron::BackwardPass() {

	double error_val = 0;

	for (int i = 0; i < this->weights.size(); i++) {
		error_val = this->neuron_error * this->weights[i];
		this->prev_layer->AddError(i, error_val);
	}

}



Layer::Layer(int size,Layer* prev_layer) {
	if(prev_layer == NULL) {
		this->input_layer = true;
	}
	else {
		this->input_layer = false;
	}

	for (int i = 0; i < size; i++) {
		if (this->input_layer) {
			Neuron n(NULL);
			this->neurons.push_back(n);
		}
		else
		{
			Neuron n(prev_layer);
			this->neurons.push_back(n);
		}
	}
}

std::vector <double> Layer::GetValues() {
	std::vector <double> vals;

	for (int i = 0; i < this->neurons.size(); i++) {
		vals.push_back(this->neurons[i].GetValue());
	}

	return vals;
}

int Layer::GetSize() {
	return this->neurons.size();
}

void Layer::PrintWeights() {
	for (Neuron& n : this->neurons) {
		n.PrintWeights();
	}
}

void Layer::ForwardPass() {
	for (Neuron& n : this->neurons) {
		n.ForwardPass();
	}
}

void Layer::BackwardPass() {
	if (input_layer) return;
	for (Neuron& n : this->neurons) {
		n.BackwardPass();
	}
}

void Layer::GradientUpdate() {
	for (Neuron& n : this->neurons) {
		n.GradientUpdate();
	}
}

void Layer::SetValue(int index, double value) {
	this->neurons[index].SetValue(value);
}

void Layer::SetInput(std::vector <double> in_vals) {
	

	for (Neuron& n : this->neurons) {
		n.SetInput(in_vals);
	}
}

void Layer::AddError(int index, double val) {
	double total_error = this->neurons[index].GetError() + val;
	this->neurons[index].SetError(total_error);
}

int main()
{
	srand(time(NULL));
	
	std::vector <double> loss_avg;
	std::vector <double> in_vector;
	double avg;

	Layer inputLayer(1, NULL);
	Layer middleLayer(5, &inputLayer);
	Layer outputLayer(1, &middleLayer);


	for (int i = 0; i < 100; i++) {
		
		double tmp = (double)(rand() % 2000) / 100.0;
		tmp = tmp - 10.0;

		//double tmp = 0;
		//std::cin >> tmp;

		for (int j = 0; j < 5; j++) {


			/*
			//input neuron takes values and calculates it (adder with weights + act. func)
			in_vector.clear();
			in_vector.push_back(tmp);
			inputLayer.SetInput(in_vector);
			*/


			//input layer/neuron has set value to input
			inputLayer.SetValue(0, tmp);


			middleLayer.ForwardPass();
			outputLayer.ForwardPass();

			double expected = func_f(tmp);
			double result = outputLayer.GetValues()[0];
			result = result * 2.0;

			double loss = pow((expected - result), 2.0);
			avg = 0;

			loss_avg.push_back(loss);
			if (loss_avg.size() > 10) {
				loss_avg.erase(loss_avg.begin());
			}
			for (double d : loss_avg) {
				avg += d;
			}
			//std::cout << "10 rolling avg loss = " << avg / 10.0 << "\n";

			/*
			std::cout << "in Layer :\n";
			inputLayer.PrintWeights();
			std::cout << "mid Layer :\n";
			middleLayer.PrintWeights();
			std::cout << "out Layer :\n";
			outputLayer.PrintWeights();
			*/

			/*
			for (double d : inputLayer.GetValues()) {
				std::cout << " " << d;
			}
			std::cout << "\n";

			for (double d : middleLayer.GetValues()) {
				std::cout << " " << d;
			}
			std::cout << "\n";

			for (double d : outputLayer.GetValues()) {
				std::cout << " " << d;
			}

			std::cout << "\n";
			*/
			std::cout << "x = " << tmp << " , f(x) = " << expected << ", net = " << result << " , loss = " << loss << "\n";
			if (loss < (double)(1 / 1000)) break;

			//backprop here
			//Backpass
			outputLayer.AddError(0, 2*(result-expected)); //set the error for the output layer
			outputLayer.BackwardPass(); //Pass error vals to middle layer
			middleLayer.BackwardPass(); //Pass error vals to input layer

			//calculate gradient
			middleLayer.GradientUpdate();
			outputLayer.GradientUpdate();

		}
		std::cout << "\n";

	}


	std::cout << "Training completed\n";
	double test_f;
	for (int g = 0; g < 10; g++) {
		std::cout << "Test " << g + 1 << " of 10\n";
		std::cin >> test_f;

		inputLayer.SetValue(0, test_f);


		middleLayer.ForwardPass();
		outputLayer.ForwardPass();

		double expected = func_f(test_f);
		double result = outputLayer.GetValues()[0];
		result = result * 2.0;
		double loss = pow((expected - result), 2.0);

		std::cout << "x = " << test_f << " , f(x) = " << expected << ", net = " << result << " , loss = " << loss << "\n";
	}
	
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
