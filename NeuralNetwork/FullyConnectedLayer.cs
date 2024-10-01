
namespace KKNeuralNetwork
{
	/// The class is designed to be part of a fully connected feedforward neural network and interacts 
	/// with other classes such as IActivation for applying activation functions and LayerLearnData 
	/// for storing intermediate computation results.
	public class FullyConnectedLayer : Layer
    {
		public FullyConnectedLayer(int nodesIn, int nodesOut, Activation.ActivationType activationType = Activation.ActivationType.Linear) : base(nodesIn, nodesOut, activationType)
		{
		}

		public override LayerLearnData ForwardPass(double[] input)
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

		public override void AdjustParameters(double learnRate)
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

