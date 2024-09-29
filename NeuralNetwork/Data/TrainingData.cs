namespace KKNeuralNetwork
{
    /// <summary>
    /// Data container for input and expected output. Used for Neural Network training.
    /// </summary>
    public struct TrainingData
    {
        public double[] data;
        public double[] expected;

        /// <summary>
        /// NOTE: Default contructor creates empty arrays!
        /// </summary>
        public TrainingData()
        {
            data = new double[0];
            expected = new double[0];
        }

        /// <param name="data">Contains array of inputs. Should be same size as the input layer of neural network.</param>
        /// <param name="expected">Contains array of expected outputs. Should be same size as the output layer of neural network.</param>
        public TrainingData(double[] data, double[] expected)
        {
            this.data = data;
            this.expected = expected;
        }
    }
}
