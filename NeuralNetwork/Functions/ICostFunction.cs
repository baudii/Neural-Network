namespace KKNeuralNetwork
{
	/// <summary>
	/// Interface for cost functions, providing methods for calculating the cost and its derivative.
	/// </summary>
	public interface ICostFunction
	{
		/// <summary>
		/// Calculates the cost between the predicted output and the expected output.
		/// </summary>
		/// <param name="output">The predicted output values.</param>
		/// <param name="expected">The expected output values.</param>
		/// <returns>The calculated cost.</returns>
		public double CalcCost(double[] output, double[] expected);

		/// <summary>
		/// Calculates the derivative of the cost function for a specific index.
		/// </summary>
		/// <param name="output">The predicted output values.</param>
		/// <param name="expected">The expected output values.</param>
		/// <param name="index">The index for which to calculate the derivative.</param>
		/// <returns>The derivative of the cost function at the given index.</returns>
		public double CalcDerivative(double[] output, double[] expected, int index);
	}
}