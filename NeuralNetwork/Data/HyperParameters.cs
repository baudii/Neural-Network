using System;

namespace KKNeuralNetwork
{
	/// <summary>
	/// Class representing hyperparameters for training a neural network.
	/// These parameters are crucial for controlling the training process and can significantly impact performance.
	/// </summary>
	public class HyperParameters
	{
		/// <summary>
		/// Starting learning rate
		/// </summary>
		public double InitialLearnRate;
		/// <summary>
		/// Rate of learning rate decay. Choose 1 if you don't need decay. Values are clamped between 1 and 255
		/// </summary>
		public double LearnRateDecay 
		{
			get => LearnRateDecay;
			set
			{
				Math.Clamp(LearnRateDecay, 1, 255);
			} 
		}
		/// <summary>
		/// Size of the batch of the training data. Used for multithreading
		/// </summary>
		public int BatchSize;
		/// <summary>
		/// Gradient decsend momentum. Formula: (prev_x + cur_x) / 2
		/// </summary>
		public double Momentum;

		/// <summary>
		/// Regularization rate of the Neural Network
		/// </summary>
		public double Regularization;

		/// <summary>
		/// Initializes a new instance of the <see cref="HyperParameters"/> class with specified or default values.
		/// </summary>
		/// <param name="initialLearnRate">Initial learning rate (default is 0.01).</param>
		/// <param name="learnRateDecay">Learning rate decay factor (default is 0.001).</param>
		/// <param name="batchSize">Batch size for training (default is 32).</param>
		/// <param name="momentum">Momentum factor (default is 0.9).</param>
		/// <param name="regularization">Regularization term (default is 0.1).</param>
		public HyperParameters(double initialLearnRate = 0.01d, double learnRateDecay = 0.001d, int batchSize = 32, double momentum = 0.9d, double regularization = 0.1d)
		{
			this.InitialLearnRate = initialLearnRate;
			this.LearnRateDecay = learnRateDecay;
			this.BatchSize = batchSize;
			this.Momentum = momentum;
			this.Regularization = regularization;
		}
	}
}