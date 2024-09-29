namespace NeuralNetwork.Data
{
    /// <summary>
    /// The LayerLearnData class is designed to store intermediate results during 
    /// the forward and backward passes through a neural network layer. This data 
    /// can be processed in parallel to optimize calculations.
    ///
    /// Key features:
    /// - Stores the raw outputs (z), which are the weighted sums of the inputs plus biases.
    /// - Stores the activated outputs (a), which are the final outputs after applying the activation function.
    /// - Stores derivative-related data (derivMemo), used during backpropagation to hold gradients or intermediate calculations.
    ///
    /// This class is particularly useful for holding data that is necessary for both the forward pass (e.g., storing outputs)
    /// and the backward pass (e.g., storing derivatives for backpropagation). By keeping these values in a separate object, 
    /// the neural network's layer can perform operations in parallel or asynchronously, improving performance and scalability.
    /// </summary>
    public class LayerLearnData
    {
        public double[] a;
        public double[] z;
        public double[] derivMemo;

        public LayerLearnData(int nodesOut)
        {
            a = new double[nodesOut];
            z = new double[nodesOut];
            derivMemo = new double[nodesOut];
        }
    }
}