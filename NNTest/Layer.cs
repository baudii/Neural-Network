
using System.Reflection.Emit;

namespace NNTest
{
    public class Layer
    {
        public double[,] w;
        public double[] b;
        public int nodesOut;
        public int nodesIn;

        public double[,] adjustedW;
        public double[] adjustedB;

        public double[,] wVelocities;
        public double[] bVelocities;

        public IActivation activation;

        public Layer(int nodesIn, int nodesOut)
        {
            this.nodesOut = nodesOut;
            this.nodesIn = nodesIn;

            w = new double[nodesOut, nodesIn];
            b = new double[nodesOut];

            adjustedW = new double[nodesOut, nodesIn];
            adjustedB = new double[nodesOut];

            wVelocities = new double[nodesOut, nodesIn];
            bVelocities = new double[nodesOut];

            activation = Activation.GetActivation(Activation.ActivationType.Linear);

            RandomizeWeights();
        }

        public void SetWeightsAndBiases(double[,] weights, double[] biases)
        {
            this.w = weights;
            this.b = biases;
        }

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

        public void RandomizeWeights()
        {
            var rand = new Random();
            for (int i = 0; i < nodesOut; i++)
            {
                for (int j = 0; j < nodesIn; j++)
                {
                    w[i, j] = 1 - rand.NextDouble();
                }
            }
        }

        public double[] Calculate(double[] input)
        {
            var a = new double[nodesOut];
            var z = new double[nodesOut];
            for (int i = 0; i < nodesOut; i++)
            {
                z[i] = b[i];
                for (int j = 0; j < nodesIn; j++)
                {
                    z[i] += w[i, j] * input[j];
                }
            }
            for (int i = 0; i < nodesOut; i++)
            {
                a[i] = activation.Activate(z, i);
            }

            return a;
        }
        public LearnData ForwardPass(double[] input)
        {
            var currentLayerData = new LearnData(nodesOut);
            for (int i = 0; i < nodesOut; i++)
            {
                currentLayerData.z[i] = b[i];
                for (int j = 0; j < nodesIn; j++)
                {
                    currentLayerData.z[i] += w[i, j] * input[j];
                }
            }
            for (int i = 0; i < nodesOut; i++)
            {
                currentLayerData.a[i] = activation.Activate(currentLayerData.z, i);
            }
            return currentLayerData;
        }
        public void ApplyNewWeightsAndBiases(double learnRate)
        {
            for (int i = 0; i < nodesOut; i++)
            {
                b[i] -= adjustedB[i] * learnRate;
                adjustedB[i] = 0;

                for (int j = 0; j < nodesIn; j++)
                {
                    w[i, j] -= adjustedW[i, j] * learnRate;
                    adjustedW[i, j] = 0;
                }
            }
        }
    }
}

