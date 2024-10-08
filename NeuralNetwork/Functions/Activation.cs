using static System.Math;

namespace KKNeuralNetwork
{
	/// <summary>
	/// Provides various activation functions commonly used in neural networks. 
	/// The functions are implemented as readonly structs for efficient performance.
	/// </summary>
	public readonly struct Activation
	{
		/// <summary>
		/// Enum representing different types of activation functions.
		/// </summary>
		public enum ActivationType
		{
			
			/// <summary>
			/// S(x) = 1 / (1 + e^-x)
			/// Sigmoid activation function, commonly used in neural networks.
			/// It maps input values to a range between 0 and 1.
			/// </summary>
			Sigmoid,

			/// <summary>
			/// Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
			/// The hyperbolic tangent activation function, commonly used in neural networks.
			/// It maps input values to a range between -1 and 1.
			/// </summary>
			TanH,

			/// <summary>
			/// ReLU(x) = MAX(0, x)
			/// Rectified Linear Unit activation function, often used in deep learning models.
			/// It outputs zero for negative values and passes positive values as is.
			/// </summary>
			ReLU,

			/// <summary>
			/// SiLU(x) = x / (1 + e^(-x))
			/// Sigmoid-Weighted Linear Unit (also called Swish) activation function.
			/// It is a smooth, non-linear function that retains small negative values, improving performance in some models.
			/// </summary>
			SiLU,

			/// <summary>
			/// Softmax(x) = e^x_i / Σ(e^x_i)
			/// Softmax activation function, used in classification tasks, especially in the output layer of neural networks.
			/// It converts raw output scores into probabilities that sum to 1.
			/// </summary>
			Softmax,

			/// <summary>
			/// Linear(x) = x
			/// Linear activation function, typically used in regression tasks or as an output activation function for certain models.
			/// It simply returns the input value without modification.
			/// </summary>
			Linear
		}

		/// <summary>
		/// Returns the corresponding activation function implementation based on the specified activation type.
		/// </summary>
		/// <param name="type">The type of activation function to return.</param>
		/// <returns>An instance of a class implementing IActivation.</returns>
		internal static IActivation GetActivation(ActivationType type)
		{
			switch (type)
			{
				case ActivationType.Sigmoid:
					return new Sigmoid();
				case ActivationType.TanH:
					return new TanH();
				case ActivationType.ReLU:
					return new ReLU();
				case ActivationType.SiLU:
					return new SiLU();
				case ActivationType.Softmax:
					return new Softmax();
				case ActivationType.Linear:
					return new Linear();
				default:
					return new Sigmoid();
			}
		}

		// Implementation of Activation functions

		internal readonly struct Sigmoid : IActivation
		{
			public double Activate(double[] z, int index)
			{
				return 1.0 / (1 + Exp(-z[index]));
			}

			public double Derivative(double[] z, int index)
			{
				double a = Activate(z, index);
				return a * (1 - a);
			}

			public ActivationType GetActivationType()
			{
				return ActivationType.Sigmoid;
			}
		}
		internal readonly struct Linear : IActivation
		{
			public double Activate(double[] z, int index)
			{
				return z[index];
			}

			public double Derivative(double[] z, int index)
			{
				return 1;
			}

			public ActivationType GetActivationType()
			{
				return ActivationType.Linear;
			}
		}

		
		internal readonly struct TanH : IActivation
		{
			public double Activate(double[] z, int index)
			{
				double e1 = Exp(-z[index]);
				double e2 = Exp(z[index]);
				var res = (e2 - e1) / (e2 + e1);
				if (res is double.NaN)
					return Sign(z[index]);
				return res;
			}

			public double Derivative(double[] z, int index)
			{
				var t = Activate(z, index);
				return 1 - t * t;
			}

			public ActivationType GetActivationType()
			{
				return ActivationType.TanH;
			}
		}


		internal readonly struct ReLU : IActivation
		{
			public double Activate(double[] z, int index)
			{
				return Max(0, z[index]);
			}

			public double Derivative(double[] z, int index)
			{
				return z[index] > 0 ? 1 : 0;
			}

			public ActivationType GetActivationType()
			{
				return ActivationType.ReLU;
			}
		}

		internal readonly struct SiLU : IActivation
		{
			public double Activate(double[] z, int index)
			{
				return z[index] / (1 + Exp(-z[index]));
			}

			public double Derivative(double[] z, int index)
			{
				double sig = 1 / (1 + Exp(-z[index]));
				return z[index] * sig * (1 - sig) + sig;
			}

			public ActivationType GetActivationType()
			{
				return ActivationType.SiLU;
			}
		}


		internal readonly struct Softmax : IActivation
		{
			public double Activate(double[] z, int index)
			{
				double expSum = 0;
				for (int i = 0; i < z.Length; i++)
				{
					expSum += Exp(z[i]);
				}

				double res = Exp(z[index]) / expSum;

				return res;
			}

			public double Derivative(double[] z, int index)
			{
				double expSum = 0;
				for (int i = 0; i < z.Length; i++)
				{
					expSum += Exp(z[i]);
				}

				double ex = Exp(z[index]);

				return (ex * expSum - ex * ex) / (expSum * expSum);
			}

			public ActivationType GetActivationType()
			{
				return ActivationType.Softmax;
			}
		}
	}
}

