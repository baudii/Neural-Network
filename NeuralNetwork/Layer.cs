using NeuralNetwork.Data;
using NeuralNetwork.Functions;

namespace NeuralNetwork
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
    ///
    /// The class is designed to be part of a fully connected feedforward neural network and interacts 
    /// with other classes such as IActivation for applying activation functions and LayerLearnData 
    /// for storing intermediate computation results.
    /// </summary>

    public class Layer
    {
        public double[,] w;
        public double[] b;
        public int nodesOut;
        public int nodesIn;

        public double[,] adjustW;
        public double[] adjustB;

        public double[,] wVelocities;
        public double[] bVelocities;

        public IActivation activation;

        public Layer(int nodesIn, int nodesOut)
        {
            this.nodesOut = nodesOut;
            this.nodesIn = nodesIn;

            w = new double[nodesOut, nodesIn];
            b = new double[nodesOut];

            adjustW = new double[nodesOut, nodesIn];
            adjustB = new double[nodesOut];

            wVelocities = new double[nodesOut, nodesIn];
            bVelocities = new double[nodesOut];

            activation = Activation.GetActivation(Activation.ActivationType.Linear);

            RandomizeWeights(true);
        }

		/// <summary>
		/// Prints all weights and biases to the console for debugging purposes.
        /// </summary>
		public void PrintWeights()
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
            for(int i = 0;i < nodesOut; i++)
            {
                Console.WriteLine("[i:" + i + "]: " + b[i]);
            }
        }

        /// <summary>
        /// Fills Layer.w array with random values in range (0,1]
        /// </summary>
        /// <param name="IsGaussian">If true, will use Gaussian normal distribution.</param>
        public void RandomizeWeights(bool IsGaussian)
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
		public LayerLearnData ForwardPass(double[] input)
        {
            var currentLayerLearnData = new LayerLearnData(nodesOut);
            for (int i = 0; i < nodesOut; i++)
            {
                currentLayerLearnData.z[i] = b[i];
                for (int j = 0; j < nodesIn; j++)
                {
                    currentLayerLearnData.z[i] += w[i, j] * input[j];
                }
                currentLayerLearnData.a[i] = activation.Activate(currentLayerLearnData.z, i);
            }
            return currentLayerLearnData;
        }


		/// <summary>
		/// Updates the weights and biases based on the adjustments calculated during backpropagation.
		/// </summary>
		/// <param name="learnRate">The learning rate to apply for weight updates.</param>
		public void AdjustParameters(double learnRate)
        {
            for (int i = 0; i < nodesOut; i++)
            {
                b[i] -= adjustB[i] * learnRate;
                adjustB[i] = 0;

                for (int j = 0; j < nodesIn; j++)
                {
                    w[i, j] -= adjustW[i, j] * learnRate;
                    adjustW[i, j] = 0;
                }
            }
        }
    }
}

