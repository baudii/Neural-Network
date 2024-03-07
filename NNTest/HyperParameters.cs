public class HyperParameters
{
    public double initialLearningRate;
    public double learnRateDecay;
    public int minibatchSize;
    public double momentum;
    public double regularization;

    public HyperParameters()
    {
        initialLearningRate = 0.0019;
        learnRateDecay = 0.04;
        minibatchSize = 32;
        momentum = 0.9;
        regularization = 0.1;
    }

}