/// <summary>
/// Class representing hyperparameters for training a neural network.
/// These parameters are crucial for controlling the training process and can significantly impact performance.
/// </summary>
public class HyperParameters
{
    public double initialLearnRate;
    public double learnRateDecay;
    public int batchSize;
    public double momentum;
    public double regularization;

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
        this.initialLearnRate = initialLearnRate;
        this.learnRateDecay = learnRateDecay;
        this.batchSize = batchSize;
        this.momentum = momentum;
        this.regularization = regularization;
    }
}