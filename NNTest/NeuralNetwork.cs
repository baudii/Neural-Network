using System.Diagnostics;

namespace NNTest
{
    public class NeuralNetwork
    {
        Stopwatch timer;
        List<Layer> layers;
        double[] initialInputs;
        ICostFunction costFunction;
        HyperParameters hyperParameters;
        double overallCost = 0;
        int iterationsElapsed = 0;
        double updateTime = 1;
        public NeuralNetwork(int inputSize, HyperParameters hyperParameters, CostFunction.CostType costFunctionType = CostFunction.CostType.MSE)
        {
            initialInputs = new double[inputSize];
            layers = new List<Layer>();
            costFunction = CostFunction.GetCostFunction(costFunctionType);
            this.hyperParameters = hyperParameters;
            timer = new Stopwatch();
            timer.Start();
        }

        public void AddLayers(Activation.ActivationType activationType, params int[] sizes)
        {
            for (int i = 0; i < sizes.Length; i++)
            {
                var layer = new Layer(GetNodesIn(), sizes[i]);
                layer.activation = Activation.GetActivation(activationType);
                layers.Add(layer);
            }

            int GetNodesIn() => layers.Count == 0 ? initialInputs.Length : layers.Last().nodesOut;
        }

        public double[] Calculate(double[] input)
        {
            for (int i = 0; i < layers.Count; i++)
                input = layers[i].Calculate(input);
            return input;
        }
        public void Learn(InputData input, double learnRate)
        {
            var learnData = ForwardPass(input.data, input.expected);
            BackPass(learnData, input.expected);

            foreach (var layer in layers)
                layer.ApplyNewWeightsAndBiases(learnRate);
        }
        public void Learn(InputData[] minibatch, double learnRate)
        {
            Parallel.For(0, minibatch.Length, (i) =>
            {
                var learnData = ForwardPass(minibatch[i].data, minibatch[i].expected);
                BackPass(learnData, minibatch[i].expected);
            });

            foreach (var layer in layers)
                layer.ApplyNewWeightsAndBiases(learnRate / minibatch.Length);
        }

        public void Learn(InputData[] batch, double learnRate, int miniBatchSize)
        {
            for (int i = 0; i < batch.Length / miniBatchSize + Math.Sign(batch.Length % miniBatchSize); i++)
            {
                var miniBatch = new InputData[Math.Min(batch.Length - i * miniBatchSize, miniBatchSize)];

                for (int j = 0; j < miniBatch.Length; j++)
                    miniBatch[j] = batch[i * miniBatchSize + j];

                Learn(miniBatch, learnRate);
            }
        }

        List<LearnData> ForwardPass(double[] input, double[] expected)
        {
            if (input.Length != initialInputs.Length)
                throw new ArgumentOutOfRangeException("Input size mismatch");
            initialInputs = input;
            List<LearnData> learnData = new List<LearnData>();

            foreach (var layer in layers)
            {
                learnData.Add(layer.ForwardPass(input));
                input = learnData.Last().a;
            }

            var cost = costFunction.CalcCost(learnData.Last().a, expected);

            PrintResults(cost, learnData.Last().a[0], expected[0]);

            return learnData;
        }

        /*void BackPass(double[] output, double[] expected, LearnData learnData)
        {
            var last = layers.Count - 1;
            for (int i = 0; i < layers[last].nodesOut; i++)
            {

                var costDeriv = costFunction.CalcDerivative(output, expected, i);
                var activationDeriv = layers[last].activation.Derivative(layers[last].z, i);
                var result = activationDeriv * costDeriv;

                //CrossEntropy + Softmax
                //var result = layers[last].a[i];
                //if (expected[i] == 1)
                //    result -= 1;

                layers[last].derivMemo[i] = result;
            }
            UpdateGradients(layers[last], layers[last - 1].a, derivMemo);

            for (int i = last - 1; i >= 0; i--)
            {
                CalculateDerivMemo(layers[i], layers[i + 1]);
                if (i == 0)
                    UpdateGradients(layers[i], initialInputs, derivMemo);
                else
                    UpdateGradients(layers[i], layers[i - 1].a, derivMemo);
            }
        }*/
        void BackPass(List<LearnData> learnDatas, double[] expected)
        {
            var last = layers.Count - 1;
            CalculateDerivMemoOutput(layers[last], learnDatas[last], expected);
            AdjustWeightsAndBiases(layers[last], learnDatas[last].derivMemo, learnDatas[last-1].a);

            double[] leftActivations;
            for (int i = last - 1; i >= 0; i--)
            {
                CalculateDerivMemoHidden(layers[i], layers[i + 1], learnDatas[i], learnDatas[i + 1].derivMemo);

                if (i > 0) leftActivations = learnDatas[i - 1].a;
                else leftActivations = initialInputs;

                AdjustWeightsAndBiases(layers[i], learnDatas[i].derivMemo, leftActivations);
            }
        }

        void CalculateDerivMemoOutput(Layer lastLayer, LearnData lastLayerLearnData, double[] expected)
        {
            for (int i = 0; i < lastLayer.nodesOut; i++)
            {
                var costDeriv = costFunction.CalcDerivative(lastLayerLearnData.a, expected, i);
                var activationDeriv = lastLayer.activation.Derivative(lastLayerLearnData.z, i);
                /*                CrossEntropy + Softmax
                                * var result = layers[last].a[i];
                                if (expected[i] == 1)
                                    result -= 1;*/

                lastLayerLearnData.derivMemo[i] = activationDeriv * costDeriv;
            }
        }

        void CalculateDerivMemoHidden(Layer currentLayer, Layer rightLayer, LearnData curLayerLearnData, double[] rightLayerDerivMemo)
        {
            for (int i = 0; i < currentLayer.nodesOut; i++)
            {
                double dCdA = 0;
                for (int j = 0; j < rightLayer.nodesOut; j++)
                {
                    dCdA += rightLayer.w[j, i] * rightLayerDerivMemo[j];
                }
                var dAdZ = currentLayer.activation.Derivative(curLayerLearnData.z, i);

                curLayerLearnData.derivMemo[i] = dCdA * dAdZ;
            }
        }

        void AdjustWeightsAndBiases(Layer layer, double[] derivMemo, double[] leftActivations)
        {
            lock (layer.adjustedW)
                for (int i = 0; i < layer.nodesOut; i++)
                    for (int j = 0; j < layer.nodesIn; j++)
                        layer.adjustedW[i, j] += derivMemo[i] * leftActivations[j];
            
            lock (layer.adjustedB)
                for (int i = 0; i < layer.nodesOut; i++)
                    layer.adjustedB[i] += derivMemo[i];
        }

        void PrintResults(double summedCost, double output, double expected)
        {
            lock (timer)
            {
                overallCost += summedCost;
                iterationsElapsed++;

                if (timer.Elapsed.TotalSeconds > updateTime)
                {
                    Console.WriteLine("Cost: " + overallCost / iterationsElapsed + "\n Output: " + output + "\n Expected: " + expected);
                    overallCost = 0;
                    iterationsElapsed = 0;
                    timer.Restart();
                }
            }
        }

        public void SaveWeights(string path)
        {
            if (!File.Exists(path))
            {
                var file = File.Create(path);
                file.Close();
            }
            List<string> saveData = new List<string>();
            string networkId = "I_" + layers.Count + "_" + initialInputs.Length;
            for (int l = 0; l < layers.Count; l++)
            {
                networkId += "_" + layers[l].nodesOut + "_" + layers[l].nodesIn;
            }
            saveData.Add(networkId);
            for (int l = 0; l < layers.Count; l++)
            {
                for (int i = 0; i < layers[l].nodesOut; i++)
                {
                    saveData.Add("b_" + l + "_" + i + "_" + layers[l].b[i] + "_" + layers[l].bVelocities[i]);
                    for (int j = 0; j < layers[l].nodesIn; j++)
                    {
                        saveData.Add(l + "_" + i + "_" + j + "_" + layers[l].w[i, j] + "_" + layers[l].wVelocities[i, j]);
                    }
                }
            }
            File.WriteAllLines(path, saveData);
        }

        public bool LoadWeights(string path)
        {
            if (!File.Exists(path))
                return false;

            var loadData = File.ReadAllLines(path);

            foreach (string line in loadData)
            {
                int l, i, j;
                var res = line.Split('_');
                
                if (res[0] == "I")
                {
                    if (initialInputs.Length != int.Parse(res[2]) || layers.Count != int.Parse(res[1]))
                        return false;

                    int charIndex = 2;
                    for (int k = 0; k < layers.Count; k++)
                    {
                        charIndex++;
                        if (layers[k].nodesOut != int.Parse(res[charIndex]))
                            return false;
                        charIndex++;
                        if (layers[k].nodesIn != int.Parse(res[charIndex]))
                            return false;
                    }
                    continue;
                }
                if (res[0] == "b")
                {
                    l = int.Parse(res[1]);
                    i = int.Parse(res[2]);
                    layers[l].b[i] = double.Parse(res[3]);
                    layers[l].bVelocities[i] = double.Parse(res[4]);
                    continue;
                }
                               
                l = int.Parse(res[0]);
                i = int.Parse(res[1]);
                j = int.Parse(res[2]);

                layers[l].w[i, j] = double.Parse(res[3]);
                layers[l].wVelocities[i, j] = double.Parse(res[4]);
            }
            return true;
        }
    }
}
