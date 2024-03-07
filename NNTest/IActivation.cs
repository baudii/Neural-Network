public interface IActivation
{
    public double Activate(double[] z, int index);

    public double Derivative(double[] a, int index);

    Activation.ActivationType GetActivationType();
}


