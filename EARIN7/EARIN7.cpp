// EARIN7.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>

#define p1 2
#define p2 4
double LEARING_RATE = 0.001;
double Max_LR = 0.001;
int repeats = 1000;
int precision = 2; //how many decimal places
int divider_prec; //used for rounding up/down
int batch_size = 10;


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

void update_LR(int iteration) {
	LEARING_RATE = Max_LR / iteration;
}

class Layer;

class Neuron
{
private:
	double neuron_value;
	double bias;
	double neuron_error;
	double previous_error;
	double previous_weight;
	double neuron_sum;
	std::vector <double> weights;
	std::vector <double> inputs;
	bool input_neuron;
	bool output_neuron;

	Layer* prev_layer;

public:
	Neuron(Layer* prev_layer_ptr, bool out_neuron);
	double GetValue();
	double GetError();
	double GetFd();
	double GetWeigth(int index);

	void PrintWeights();

	void ForwardPass();
	void BackwardPass();
	void SetValue(double val);
	void SetInput(std::vector <double> in_vals);
	void SetError(double error_val);
	void PassError(double error_val, double weight);
	void GradientUpdate();
};

class Layer
{
private:
	std::vector <Neuron> neurons;
	bool input_layer;
	bool output_layer;


public:
	Layer(int size,Layer* prev_layer, bool outputlayer);
	std::vector <double> GetValues();
	int GetSize();
	double GetWeight(int index, int weight_index);
	double GetError(int index);
	double GetFd(int index);


	void PrintWeights();

	void ForwardPass();
	void BackwardPass();
	void SetValue(int index, double value);
	void SetError(int index, double error_value);
	void SetInput(std::vector <double> in_vals);
	void AddError(int index, double val);
	void GradientUpdate();
};

Neuron::Neuron(Layer* prev_layer_ptr,bool out_neuron) {

	int prev_layer_size;

	if (prev_layer_ptr == NULL) {
		prev_layer_size = 0;
		this->input_neuron = true;
	}
	else {
		prev_layer_size = prev_layer_ptr->GetSize();
		this->input_neuron = false;
	}

	if (out_neuron) {
		this->output_neuron = true;
	}
	else {
		this->output_neuron = false;
	}

	
	//set connections
	for (int i = 0; i < prev_layer_size; i++) {
		this->prev_layer = prev_layer_ptr;
	}
	this->neuron_error = 0;

	//set random bias
	int rand_int = rand() % 2000;
	this->bias = (double)(rand_int / 1000.0);
	this->bias -= 1.0;

	//set random weights
	for(int i = 0; i < prev_layer_size; i++) {
		int rand_int = rand() % 2000;
		double rand_float = rand_int / 1000.0;
		rand_float -= 1.0;
		this->weights.push_back(rand_float);
	}
}

double Neuron::GetValue() {
	return this->neuron_value;
}

double Neuron::GetError() {
	return this->neuron_error;
}

double Neuron::GetFd() {
	return sigmoid_derivative(this->neuron_sum);
}

double Neuron::GetWeigth(int index) {
	return this->weights[index];
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

void Neuron::PassError(double error_val, double weight) {
	this->previous_error = error_val;
	this->previous_weight = weight;
}

void Neuron::GradientUpdate() {
	double grad;

	//for each weigth
	for (int i = 0; i < this->weights.size();i++) {
		grad = this->inputs[i] * sigmoid_derivative(this->neuron_sum) * this->neuron_error * LEARING_RATE;
		//update weight
		this->weights[i] -= grad;
	}

	//update bias
	grad = sigmoid_derivative(this->neuron_sum) * this->neuron_error;

	this->bias -= grad;
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
	

	if (this->output_neuron) {
		this->neuron_value = (totalVal);
	}
	else {
		this->neuron_value = sigmoid(totalVal);
	}

	
	
	

}

//Pass error to prev layer
void Neuron::BackwardPass() {
	
	if (this->input_neuron) {
		return;
	}

	double error_val = 0;

	for (int i = 0; i < this->weights.size(); i++) {
		error_val = this->neuron_error * sigmoid_derivative(this->neuron_sum) * this->weights[i];

		this->prev_layer->AddError(i, error_val);
	}

	
	/*
	//for each of weights coming into the neuron
	for (int i = 0; i < this->weights.size(); i++) {
		//d_cost / d_weight = d_input / d_weights * d_out / d_input * d_Cost / d_output = A * B * C
		//double A = input[i];
		//double B = sigmoid_derivative(this->neuron_value);

		//double C = this->neuron_error; 
		//if output layer : neuron_error = 2(output - expected)
		//else : sigmoid_derivative(this->neuron_value)*2(out-expected)*weight 


	}
	*/
}





Layer::Layer(int size,Layer* prev_layer, bool outputlayer) {
	if(prev_layer == NULL) {
		this->input_layer = true;
	}
	else {
		this->input_layer = false;
	}

	if (outputlayer) {
		this->output_layer = true;
	}
	else {
		this->output_layer = false;
	}

	for (int i = 0; i < size; i++) {
		if (this->input_layer) {
			Neuron n(NULL,this->output_layer);
			this->neurons.push_back(n);
		}
		else
		{
			Neuron n(prev_layer,this->output_layer);
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

double Layer::GetWeight(int index, int weight_index) {
	return this->neurons[index].GetWeigth(weight_index);
}

double Layer::GetError(int index) {
	return this->neurons[index].GetError();
}

double Layer::GetFd(int index) {
	return this->neurons[index].GetFd();
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

void Layer::SetError(int index, double error_value) {
	this->neurons[index].SetError(error_value);
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
	std::cout.precision(4);
	divider_prec = pow(10, precision);
	
	std::vector <double> loss_avg;
	std::vector <double> in_vector;
	double tmp;

	Layer inputLayer(1, NULL, false);
	Layer middleLayer(5, &inputLayer, false);
	Layer outputLayer(1, &middleLayer, true);

	for (int i = 0; i < repeats*5; i++) {
		
		double expected;
		double result;
		double loss;

		/*
		loss_avg.clear();
		for (int j = 0; j < batch_size; j++) {

			if (i % 5 == 0) {
			//Select random value from the domain
			tmp = (double)(rand() % 2000) / 100.0;
			//round to x decimal places
			tmp = roundf(tmp * divider_prec) / divider_prec;
			//tmp = -6.4;
			tmp = tmp - 10.0;
			}

			//Select random value from the domain
			tmp = (double)(rand() % 2000) / 100.0;
			//round to x decimal places
			tmp = roundf(tmp * divider_prec) / divider_prec;
			//tmp = -6.4;
			tmp = tmp - 10.0;

			inputLayer.SetValue(0, tmp);

			middleLayer.ForwardPass();
			outputLayer.ForwardPass();

			//Calculate expected, result and loss
			expected = func_f(tmp);
			result = outputLayer.GetValues()[0];
			//result = result * 2.0;

			loss = pow((result - expected), 2.0);
			loss_avg.push_back(loss);
		}
		*/
		
		
		//Select random value from the domain
		tmp = (double)(rand() % 2000) / 100.0;
		//round to x decimal places
		tmp = roundf(tmp * divider_prec) / divider_prec;
		//tmp = -6.4;
		tmp = tmp - 10.0;
		
		 
		//Calculate the network estimate for given input
		//input layer/neuron has set value to input
		inputLayer.SetValue(0, tmp);

		middleLayer.ForwardPass();
		outputLayer.ForwardPass();

		//Calculate expected, result and loss
		expected = func_f(tmp);
		result = outputLayer.GetValues()[0];
		//result = result * 2.0;

		loss = pow((result - expected), 2.0);

		//print all
		if (!(i % 100)) {
			std::cout << "x = " << tmp << " , f(x) = " << expected << ", net = " << result << " , loss = " << loss << "\n";
		}
		
		//Training part
		//for quadratic loss function f' = 2(result-expected)
		outputLayer.AddError(0, 2 * (result - expected)); //set the error for the output layer
		outputLayer.BackwardPass(); //Pass error vals to middle layer

		//calculate gradient and update weights
		outputLayer.GradientUpdate();
		middleLayer.GradientUpdate();

	}

	std::cout << "Training completed\n";
	double test_f;
	for (int g = 0; g < 10; g++) {
		std::cout << "Test " << g + 1 << " of 10\n";
		test_f = (double)(rand() % 2000) / 100.0;
		test_f = roundf(test_f * divider_prec) / divider_prec;
		test_f = test_f - 10.0;
		//std::cin >> test_f;

		inputLayer.SetValue(0, test_f);


		middleLayer.ForwardPass();
		outputLayer.ForwardPass();

		double expected = func_f(test_f);
		double result = outputLayer.GetValues()[0];
		double loss = pow((result - expected), 2.0);

		std::cout << "x = " << test_f << " , f(x) = " << expected << ", net = " << result<< " , loss = " << loss << "\n";
	}
	
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu