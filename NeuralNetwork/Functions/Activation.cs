using static System.Math;

namespace NeuralNetwork.Functions
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
            Sigmoid,
            TanH,
            ReLU,
            SiLU,
            Softmax,
            Linear
        }

        /// <summary>
        /// Returns the corresponding activation function implementation based on the specified activation type.
        /// </summary>
        /// <param name="type">The type of activation function to return.</param>
        /// <returns>An instance of a class implementing IActivation.</returns>
        public static IActivation GetActivation(ActivationType type)
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

        public readonly struct Sigmoid : IActivation
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
        public readonly struct Linear : IActivation
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


        public readonly struct TanH : IActivation
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


        public readonly struct ReLU : IActivation
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

        public readonly struct SiLU : IActivation
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


        public readonly struct Softmax : IActivation
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

