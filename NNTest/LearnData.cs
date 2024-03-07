namespace NNTest
{
    public class LearnData
    {
        public double[] a;
        public double[] z;
        public double[] derivMemo;

        public LearnData(int nodesOut)
        {
            a = new double[nodesOut];
            z = new double[nodesOut];
            derivMemo = new double[nodesOut];
        }
    }
}