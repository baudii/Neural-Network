namespace NeuralNetwork.Functions
{
    /// <summary>
    /// Interface representing an activation function in a neural network.
    /// Classes implementing this interface must provide methods for activation and derivative calculations.
    /// </summary>
    public interface IActivation
    {
        /// <summary>
        /// Activates the input value based on the specific activation function implementation.
        /// </summary>
        /// <param name="z">An array of input values to activate.</param>
        /// <param name="index">The index of the value to activate.</param>
        /// <returns>The activated value at the specified index.</returns>
        public double Activate(double[] z, int index);

        /// <summary>
        /// Calculates the derivative of the activation function at the specified index for backpropagation.
        /// </summary>
        /// <param name="a">An array of activated values (outputs) from the previous layer.</param>
        /// <param name="index">The index of the value for which to calculate the derivative.</param>
        /// <returns>The derivative of the activation function at the specified index.</returns>
        public double Derivative(double[] a, int index);

        /// <summary>
        /// Gets the type of activation function implemented by this instance.
        /// </summary>
        /// <returns>The corresponding activation type from the Activation.ActivationType enumeration.</returns>
        Activation.ActivationType GetActivationType();
    }
}
