# Neural Network Library
## Overview

KKNeuralNetwork is a simple neural network library designed for a variety of machine learning tasks. While it currently supports only fully connected (dense) layers, future updates may include support for Convolutional, Deconvolutional, and Recurrent layers. The library offers the following activation functions:

    Sigmoid, TanH, ReLU, SiLU, Linear

For cost functions, it supports:

    Mean Squared Error (MSE)
    Mean Squared Logarithmic Error (MSLE)
    Mean Absolute Percentage Error (MAPE)

Softmax combined with Cross Entropy is also present, but the functionality is currently not operational.
## Features:

Fully connected layers: Add as many fully connected layers as needed.

Flexible cost and activation functions: Choose the appropriate functions for your neural network model.

Easy to extend: You can add more activation and cost functions based on the current architecture.

## Installation
### Visual Studio:

1) Clone the repository to your local machine using the command:
2) Open your Visual Studio project and navigate to Solution Explorer.
3) Right-click on your solution and choose Add > Reference.
4) In the Reference Manager window, click on Browse and locate the KKNeuralNetwork.dll file located in the root folder of the repository.
5) Click OK to add the reference to your project.

### Manual:

1) Download the KKNeuralNetwork.dll and KKNeuralNetwork.xml from the root folder of the repository.
2) In your .NET 6.0 project, manually add a reference to the KKNeuralNetwork.dll. You can do this by including the following line in your project file (.csproj):


```xml
<ItemGroup>
  <Reference Include="KKNeuralNetwork">
    <HintPath>path/to/KKNeuralNetwork.dll</HintPath>
  </Reference>
</ItemGroup>
```
Alternatively, if you're building the project from the command line, you can compile your project using the -r flag to include the DLL, like so:

```bash
 csc -r:KKNeuralNetwork.dll Program.cs
```
Requirements

    .NET 6.0 or higher

## Example Usage

Here's a sample program that demonstrates how to train and test a neural network using the XOR function.
```csharp
using KKNeuralNetwork;

internal class Program
{
    static string path = "YOUR_PATH_HERE";
    static string fileName = "XOR_Relu_2_4_4_1_MSLE.txt";
    static NeuralNetwork nn;

    static void Main(string[] args)
    {
        CreateNeuralNetworkXOR();
        XOR_Train();
        XOR_Test();
    }

    static void CreateNeuralNetworkXOR()
    {
        nn = new NeuralNetwork(2, CostFunction.CostType.MSLE);
        nn.AddLayers(Activation.ActivationType.ReLU, 4, 4);
        nn.AddLayers(Activation.ActivationType.Linear, 1);
        nn.LoadWeights(path + fileName);
    }

    static void XOR_Train()
    {
        Random rand = new Random();
        int dataSetSize = 3000;

        for (int i = 0; i < 10000; i++)
        {
            TrainingData[] dataSet = new TrainingData[dataSetSize];
            for (int j = 0; j < dataSetSize; j++)
            {
                int x = (int)Math.Round(rand.NextDouble());
                int y = (int)Math.Round(rand.NextDouble());
                var input = new TrainingData(new double[] { x, y }, new double[] { x ^ y });
                dataSet[j] = input;
            }
            nn.Learn(dataSet, 0.001);
        }
        nn.SaveWeights(path + fileName);
    }

    static void XOR_Test()
    {
        string a_read = Console.ReadLine();
        string b_read = Console.ReadLine();
        if (!int.TryParse(a_read, out int a) || !int.TryParse(b_read, out int b))
        {
            Console.WriteLine("Couldn't parse input");
        }
        else
        {
            double[] result = nn.Calculate(new double[] { a, b });
            Console.WriteLine(result[0]);
        }
    }
}
```
#### Explanation:

Creating the Neural Network:
  - The CreateNeuralNetworkXOR method initializes a neural network with 2 input nodes and a cost function of type MSLE.
  - It adds two hidden layers with 4 nodes each using the ReLU activation function and one output layer with the Linear activation function.
  - It then loads the saved weights from the file (if any).

Training the Network:
  - The XOR_Train method generates random XOR training data and uses it to train the network over 10,000 iterations.
  - After training, the weights are saved to a file.

Testing the Network:
  - The XOR_Test method takes two integers from the user, feeds them into the trained neural network, and prints the result.

### Project Structure

Here is a basic overview of the project's structure:
```
ROOT
├── NeuralNetwork/                  # Source code
│   ├── Data/
│   │   ├── Hyperparameters.cs      # Optional hyperparameters class for network tuning
│   │   ├── LayerLearnData.cs       # Class to store layer-specific learning data
│   │   └── TrainingData.cs         # Input/expected data passed to the network
│   ├── Functions/
│   │   ├── Activation.cs           # All activation functions implemented here
│   │   ├── IActivation.cs          # Interface for activations (Activate(), Derivative(), GetActivationType())
│   │   ├── CostFunction.cs         # All cost functions implemented here
│   │   └── ICostFunction.cs        # Interface for cost functions (CalcCost(), CalcDerivative())
│   ├── KKNeuralNetwork.csproj      # Project file for .NET
│   ├── KKNeuralNetwork.sln         # Solution file
│   ├── Layer.cs                    # Layer class containing data and methods for each layer
│   └── NeuralNetwork.cs            # Main NeuralNetwork class
├── KKNeuralNetwork.dll             # Precompiled library
├── KKNeuralNetwork.xml             # XML documentation for the library
├── .gitattributes
├── .gitignore
└── LICENSE
```

### License

This project is licensed under the MIT License. See the LICENSE file for more details.
