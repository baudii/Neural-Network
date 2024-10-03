using System;

namespace KKNeuralNetwork
{
	/// <summary>
	/// The Layer class represents a single layer in a neural network, containing nodes (neurons), 
	/// weights connecting this layer to the previous one, and biases for each node. It supports 
	/// initializing weights, applying forward passes (computing the layer's output given inputs), 
	/// and adjusting weights and biases based on backpropagation results.
	///
	/// Key features:
	/// - Holds the weight matrix (w) and bias vector (b) for connections to the previous layer.
	/// - Supports random initialization of weights, with an option to use Gaussian distribution.
	/// - Allows for computation of the weighted sum (z) and application of an activation function 
	///   to get the final output (a) during the forward pass.
	/// - Provides methods to print the current weights and biases for debugging purposes.
	/// - Updates weights and biases based on gradient adjustments calculated during training, 
	///   using the AdjustParameters method, which applies the learning rate.
	/// - Optionally supports advanced optimization techniques through weight and bias velocities.
	/// </summary>
	internal abstract class Layer
	{
		internal double[,] w;
		internal double[] b;
		internal int nodesOut;
		internal int nodesIn;

		internal double[,] adjustW;
		internal double[] adjustB;

		internal double[,] wVelocities;
		internal double[] bVelocities;

		internal IActivation activation;


		internal Layer(int nodesIn, int nodesOut, Activation.ActivationType activationType = Activation.ActivationType.Linear)
		{
			this.nodesOut = nodesOut;
			this.nodesIn = nodesIn;

			w = new double[nodesOut, nodesIn];
			b = new double[nodesOut];

			adjustW = new double[nodesOut, nodesIn];
			adjustB = new double[nodesOut];

			wVelocities = new double[nodesOut, nodesIn];
			bVelocities = new double[nodesOut];

			activation = Activation.GetActivation(activationType);

			RandomizeWeights(true);
		}

		/// <summary>
		/// Prints all weights and biases to the console for debugging purposes.
		/// </summary>
		internal void PrintWeights()
		{
			Console.WriteLine("Weights:");
			for (int i = 0; i < nodesOut; i++)
			{
				for (int j = 0; j < nodesIn; j++)
				{
					Console.WriteLine("[i:" + i + "; j:" + j + "] " + w[i, j]);
				}
			}
			Console.WriteLine("Biases");
			for (int i = 0; i < nodesOut; i++)
			{
				Console.WriteLine("[i:" + i + "]: " + b[i]);
			}
		}

		/// <summary>
		/// Fills Layer.w array with random values in range (0,1]
		/// </summary>
		/// <param name="IsGaussian">If true, will use Gaussian normal distribution.</param>
		internal void RandomizeWeights(bool IsGaussian)
		{
			var rand = new Random();
			for (int i = 0; i < nodesOut; i++)
			{
				for (int j = 0; j < nodesIn; j++)
				{
					double res = 1.0 - rand.NextDouble();
					if (IsGaussian)
					{
						double u2 = 1.0 - rand.NextDouble();
						res = Math.Sqrt(-2.0 * Math.Log(res)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
					}
					w[i, j] = res;
				}
			}
		}
		/// <summary>
		/// Passes given input through every node of this layer. 
		/// Creates LayerLearnData object and fills:
		/// LayerLearnData.z (output after multiplying with weights and adding bias)
		/// LayerLearnData.a (output after activation funcction applied)
		/// </summary>
		/// <param name="input">Size of this array must equal nodesIn of this layer</param>
		/// <returns>LayerLearnData object</returns>
		internal abstract LayerLearnData ForwardPass(double[] input);

		/// <summary>
		/// Updates the weights and biases based on the adjustments calculated during backpropagation.
		/// </summary>
		/// <param name="learnRate">The learning rate to apply for weight updates.</param>
		internal abstract void ApplyChanges(double learnRate);
	}
}
