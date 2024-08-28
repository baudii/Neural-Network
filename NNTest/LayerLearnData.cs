namespace NNTest
{
    /// <summary>
    /// This data is stored in a separate class for Parallel calculations
    /// </summary>
    public class LayerLearnData
    {
        public double[] a;
        public double[] z;
        public double[] derivMemo;

        public LayerLearnData(int nodesOut)
        {
            a = new double[nodesOut];
            z = new double[nodesOut];
            derivMemo = new double[nodesOut];
        }
    }
}