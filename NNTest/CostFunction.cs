﻿/// <summary>
/// Provides various cost functions for evaluating the difference between predicted
/// output and the expected output in machine learning models. The cost functions are
/// implemented as readonly structs for efficient performance.
/// </summary>
public readonly struct CostFunction
{
    public enum CostType
    {
        MSE, // Mean Squared Error (MSE)
		CrossEntropy, // Cross-Entropy
		MSLE, // Mean Squared Logarithmic Error (MSLE)
		MAPE // Mean Absolute Percentage Error (MAPE)
	}

	/// <summary>
	/// Returns the corresponding cost function implementation based on the specified cost type.
	/// </summary>
	/// <param name="costType">The type of cost function to return.</param>
	/// <returns>An instance of a class implementing ICostFunction.</returns>
	public static ICostFunction GetCostFunction(CostType costType)
    {
        switch (costType)
        {
            case CostType.MSE:
                return new MSE();
            case CostType.CrossEntropy:
                return new CrossEntropy();
            case CostType.MSLE:
                return new MSLE();
            case CostType.MAPE: 
                return new MAPE();
            default:
                throw new ArgumentException();
        }
    }

	public readonly struct MAPE : ICostFunction
    {
        public double CalcCost(double[] output, double[] expected)
        {
            double error = 0;
            for (int i = 0; i < output.Length; i++)
            {
                error += Math.Abs((expected[i] - output[i]) / (expected[i] + double.Epsilon));
            }
            return error / output.Length;
        }

        public double CalcDerivative(double[] output, double[] expected, int index)
        {
            var a = output[index];
            var e = expected[index];
            return (a / (e * e) - 1 / e) / CalcCost(output, expected);
        }
    }


    public readonly struct MSLE : ICostFunction
    {
        public double CalcCost(double[] output, double[] expected)
        {
            double error = 0;
            for (int i = 0; i < output.Length; i++)
            {
                error += Math.Pow(Math.Log(output[i] + 1) - Math.Log(expected[i] + 1), 2);
            }
            return error;
        }

        public double CalcDerivative(double[] output, double[] expected, int index)
        {
            return 2 * (Math.Log(output[index] + 1) - Math.Log(expected[index] + 1)) / (output[index] + 1);
        }
    }

    public readonly struct MSE : ICostFunction
    {
        public double CalcCost(double[] output, double[] expected)
        {
            double error = 0;
            for (int i = 0; i < output.Length; i++)
            {
                error += Math.Pow(output[i] - expected[i], 2);
            }
            return error / output.Length;
        }

        public double CalcDerivative(double[] output, double[] expected, int index)
        {
            return 2 * (output[index] - expected[index]) / output.Length;
        }
    }

    public readonly struct CrossEntropy : ICostFunction
    {
        public double CalcCost(double[] output, double[] expected)
        {
            for (int i = 0; i < expected.Length; i++)
            {
                if (expected[i] == 1)
                {
                    return -Math.Log(output[i]);
                }
            }
            return 1;
        }

        public double CalcDerivative(double[] output, double[] expected, int index)
        {
            if (expected[index] == 0)
                return 1;
            return -1 / output[index];
        }
    }
}
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