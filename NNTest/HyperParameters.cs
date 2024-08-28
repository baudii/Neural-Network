public class HyperParameters
{
    public double initialLearnRate;
    public double learnRateDecay;
    public int batchSize;
    public double momentum;
    public double regularization;

    public HyperParameters(double initialLearnRate = 0.01d, double learnRateDecay = 0.001d, int batchSize = 32, double momentum = 0.9d, double regularization = 0.1d)
    {
        this.initialLearnRate = initialLearnRate;
        this.learnRateDecay = learnRateDecay;
        this.batchSize = batchSize;
        this.momentum = momentum;
        this.regularization = regularization;
    }
}