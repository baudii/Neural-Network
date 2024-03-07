using System.Reflection.Metadata.Ecma335;

public static class Func
{
    public static double SiLU(double[] inputs, int index)
    {
        return inputs[index] / (1 + Math.Exp(-inputs[index]));
    }

    public static double SiLUDerivative(double[] inputs, int index)
    {
        double sig = 1 / (1 + Math.Exp(-inputs[index]));
        return inputs[index] * sig * (1 - sig) + sig;
    }
    public static double SoftMax(double[] values, int index)
    {
        double sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sum += Math.Exp(values[i]);
        }
        return Math.Exp(values[index]) / sum;
    }
    public static double SoftMaxDerivative(double[] inputs, int index)
    {
        double expSum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            expSum += Math.Exp(inputs[i]);
        }

        double ex = Math.Exp(inputs[index]);
        return (ex * expSum - ex * ex) / (expSum * expSum);
    }

    public static double Sigmoid(double z)
    {
        return 1 / (1 + Math.Exp(-z));
    }

    public static double SigmoidDerivative(double sigmoid)
    {
        return sigmoid * (1 - sigmoid);
    }

    public static double ReLU(double z)
    {
        return Math.Max(0, z);
    }

    public static double ReLuDerivative(double z)
    {
        return z >= 0 ? 1 : 0;
    }

    public static double MSE(double[] outputs, double[] expected)
    {
        double error = 0;
        for (int i = 0; i < outputs.Length; i++)
        {
            error += Math.Pow(outputs[i] - expected[i], 2);
        }
        return error;
    }

    public static double CrossEntropyCost(double[] outputs, double[] expected)
    {
        for(int i = 0; i < expected.Length; i++)
        {
            if (expected[i] == 1)
            {
                return -Math.Log(outputs[i]);
            }
        }
        return 1;
    }

    public static double CrossEntropyDerivative(double[] output, double[] expected, int index)
    {
        if (expected[index] == 0)
            return 1;
        return - 1 / output[index];
    }

    public static double MSEDerivative(double[] output, double[] expected, int index)
    {
        return 2 * (output[index] - expected[index]) / output.Length;
    }

    public static double[] Normalize(double[] data)
    {
        var sum = Enumerable.Sum(data);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = data[i] / sum;
        }
        return data;
    }
    public static double[] CreateOneHot(int index, int num)
    {
        double[] oneHot = new double[num];
        oneHot[index] = 1;
        return oneHot;
    }
}
