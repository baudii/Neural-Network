
using System.Reflection.Emit;

namespace NNTest
{
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

        public void RandomizeWeights(bool Gaussian)
        {
            var rand = new Random();
            for (int i = 0; i < nodesOut; i++)
            {
                for (int j = 0; j < nodesIn; j++)
                {
                    double res = 1.0 - rand.NextDouble();
                    if (Gaussian)
                    {
                        double u2 = 1.0 - rand.NextDouble();
                        res = Math.Sqrt(-2.0 * Math.Log(res)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                    }
                    w[i, j] = res;
                }
            }
        }
        // Passing given inputs through every node of this layer.
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
        /// Apply changed values. Called after ForwardPass and BackPass methods
        /// </summary>
        /// <param name="learnRate">Use small number for better approximation</param>
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

