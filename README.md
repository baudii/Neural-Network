Prerequisites

Before you start, ensure you have the following installed:

    .NET SDK (version 5.0 or later)
    A suitable IDE (e.g., Visual Studio, Visual Studio Code)

Project Structure

The project is organized as follows:

arduino

/NNTest
│
├── Activation.cs         // Activation functions implementation
├── CostFunction.cs       // Cost functions implementation
├── Layer.cs              // Layer class for the neural network
├── HyperParameters.cs     // Class for hyperparameter configuration
├── FenEvalDBHandler.cs    // Database handler for fetching FEN strings
└── Program.cs            // Main entry point

Installation

    Clone the repository:

    bash

git clone https://github.com/yourusername/yourrepository.git
cd yourrepository

Open the project in your IDE.

Restore dependencies:

bash

    dotnet restore

Creating a Neural Network

You can create a neural network by instantiating the Layer class and specifying the number of input and output nodes.
Example

csharp

using NNTest;

public class Program
{
    public static void Main(string[] args)
    {
        // Create a neural network layer with 3 input nodes and 2 output nodes
        Layer layer = new Layer(nodesIn: 3, nodesOut: 2);
        
        // Print initialized weights and biases
        layer.PrintWeights();
    }
}

Training the Neural Network

To train the neural network, you will need to pass your input data and expected output to the ForwardPass method and then adjust the parameters using the AdjustParameters method after backpropagation.
Example

csharp

double[] inputData = new double[] { 0.5, 0.3, 0.2 };
double[] expectedOutput = new double[] { 0.7, 0.1 };

// Forward pass
LayerLearnData learnData = layer.ForwardPass(inputData);

// Backward pass (implement backpropagation here)
// ...

// Adjust parameters
layer.AdjustParameters(0.01); // Using a learning rate of 0.01

Evaluating the Model

After training, you can evaluate the model using various metrics provided in the CostFunction class.
Example

csharp

double[] output = learnData.a; // Get output from the last layer
double[] expected = expectedOutput;

double cost = CostFunction.GetCostFunction(CostFunction.CostType.MSE).CalcCost(output, expected);
Console.WriteLine($"Cost: {cost}");

Adjusting Hyperparameters

You can adjust hyperparameters such as learning rate, momentum, and batch size using the HyperParameters class.
Example

csharp

HyperParameters hyperParams = new HyperParameters(initialLearnRate: 0.01, momentum: 0.9);

Example Use Cases
Use Case 1: Classification

    Prepare your dataset.
    Initialize the neural network layers according to your input features and classes.
    Train using labeled data.
    Evaluate the model on test data.

Use Case 2: Regression

    Load your dataset.
    Define the network structure to predict continuous values.
    Train the network using your input-output pairs.
    Assess model performance using relevant metrics.

Contributing to the Project

We welcome contributions! Feel free to open issues or submit pull requests to improve the project. Please refer to the CONTRIBUTING.md file for guidelines.
License

This project is licensed under the [Your Chosen License]. Please see the LICENSE file for details.
