namespace NNTest
{
    public struct InputData
    {
        public double[] data;
        public double[] expected;

        public InputData(double[] data, double[] expected)
        {
            this.data = data;
            this.expected = expected;
        }
    }
}
