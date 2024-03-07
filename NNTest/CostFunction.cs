public readonly struct CostFunction
{
    public enum CostType
    {
        MSE,
        CrossEntropy,
        MSLE,
        MAPE
    }

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

public interface ICostFunction
{
    public double CalcCost(double[] output, double[] expected);

    public double CalcDerivative(double[] output, double[] expected, int index);
}