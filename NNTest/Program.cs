
namespace NNTest
{
    public class Program
    {
        static Dictionary<char, int> pieceTypes = new Dictionary<char, int>() { { 'p', 0 }, { 'n', 1 }, { 'b', 2 }, { 'r', 3 }, { 'q', 4 }, { 'k', 5 }, };
        static double currentLearnRate;
        static HyperParameters hyperParameters = new HyperParameters();

        public static string path = "C:\\Users\\Dp\\Desktop\\Weights\\";


        static void Main(string[] args)
        {
            ChessTrainer(0);
            //Square();
            //XOR();
        }

        static void ChessTrainer(int index)
        {
            hyperParameters = new HyperParameters();
            var fileName = "ChessWeights.txt";
            Random rand = new Random();
            currentLearnRate = hyperParameters.initialLearningRate;
            var nn = new NeuralNetwork(64 * 2 + 6, hyperParameters, CostFunction.CostType.MSE);
            nn.AddLayers(Activation.ActivationType.TanH, 64, 32);
            nn.AddLayers(Activation.ActivationType.Linear, 1);
            if (!nn.LoadWeights(path + fileName))
                Console.WriteLine("Weights were not loaded");

            var fenEvals = FenEvalDBHandler.GetFenStrings(18000, 3000);
            InputData[] inputData = new InputData[fenEvals.Length];
            int b = 0;
            foreach (var item in fenEvals)
            {
                var desipheredFen = DecipherFen(item.fen);

                inputData[b] = new InputData(desipheredFen, new double[] { item.eval });
                b++;
            }
            for (int epoch = 0; epoch < 20; epoch++)
            {
                for (int i = 0; i < 40; i++)
                {
                    nn.Learn(inputData, currentLearnRate, 32);
                }

                inputData = Shuffle(rand, inputData);
                EpochCompleted(epoch);
                nn.SaveWeights(path + fileName);
            }


/*            var fenEvals = FenEvalDBHandler.GetFenStrings(32 * index + 1, 400);
            for (int epoch = 0; epoch < 100000; epoch++)
            {
                foreach (var item in fenEvals)
                {
                    var learnData = new LearnData(DecipherFen(item.fen), new double[] { item.eval });
                    nn.Learn(learnData, 0.9);
                }
            }*/
            
        }

        static double[] DecipherFen(string fen)
        {
            int cell = 0;
            double[] inputData = new double[64 * 2 + 6]; //(cells * (pieces + enPassant) * sides + whose turn is it + castles  + 50moveCounter
            Dictionary<char, double> pieceValue = new Dictionary<char, double> { { 'p', 0.1 }, { 'n', 0.31 }, { 'b', 0.32 }, { 'r', 0.5 }, { 'q', 0.9 }, { 'k', 1 }, };
            int spaceCount = 0;

            int column = 0;
            for (var i = 0; i < fen.Length; i++)
            {
                if (fen[i] == '/')
                    continue;

                if (fen[i] == ' ')
                {
                    spaceCount++;
                    continue;
                }

                if (char.IsDigit(fen[i]) && spaceCount == 0)
                {
                    cell += int.Parse(fen[i].ToString());
                    continue;
                }

                if (spaceCount == 0)
                {
                    int multiplier = 1;
                    if (char.IsLower(fen[i]))
                        multiplier = -1;
                    double value = pieceValue.GetValueOrDefault(char.ToLower(fen[i]));
                    inputData[cell] = value * multiplier;
                    cell++;
                }

                if (spaceCount == 1)
                {
                    if (fen[i] == 'w')
                        inputData[128] = 1;
                    else
                        inputData[128] = -1;
                    continue;
                }

                if (spaceCount == 2)
                {
                    if (fen[i] == '-')
                        continue;

                    int multiplier = 1;
                    if (char.IsLower(fen[i]))
                        multiplier = -1;

                    if (char.ToLower(fen[i]) == 'k')
                        inputData[131 - multiplier - 1] = 1;
                    else if (char.ToLower(fen[i]) == 'q')
                        inputData[132 - multiplier - 1] = 1;

                }


                if (spaceCount == 3)
                {
                    if (fen[i] == '-')
                        continue;

                    if (char.IsDigit(fen[i]))
                    {
                        int row = int.Parse(fen[i].ToString()) - 1;
                        inputData[64 + row * column] = 1;
                    }
                    else
                    {
                        column = (fen[i] % 32) - 1;
                    }
                }
                if (spaceCount == 4)
                {
                    inputData[133] = int.Parse(fen[i].ToString());
                    break;
                }
            }
            return inputData;
        }
        /*static double[] DecipherFen(string fen)
        {
            int cell = 0;
            double[] inputData = new double[64 * 7 * 2 + 2 + 4 + 1]; //(cells * (pieces + enPassant) * sides + whose turn is it + castles  + 50moveCounter
            Dictionary<char, int> pieceIndex = new Dictionary<char, int> { { 'p', 0 }, { 'n', 1 }, { 'b', 2 }, { 'r', 3 }, { 'q', 4 }, { 'k', 5 }, };
            int spaceCount = 0;

            int row = 0;
            int column = 0;
            for (var i = 0; i < fen.Length; i++)
            {
                if (fen[i] == '/')
                    continue;

                if (char.IsDigit(fen[i]) && spaceCount == 0)
                {
                    cell += int.Parse(fen[i].ToString());
                    continue;
                }

                if (fen[i] == ' ')
                {
                    spaceCount++;
                    continue;
                }

                if (spaceCount == 0)
                {
                    int multiplier = 0;
                    if (char.IsLower(fen[i]))
                        multiplier = 1;
                    int index = cell * 7 + 64 * 7 * multiplier + pieceIndex.GetValueOrDefault(char.ToLower(fen[i])) + 3 * multiplier + 3;
                    inputData[index] = 1;
                    cell++;
                }

                if (spaceCount == 1)
                {
                    if (fen[i] == 'w')
                        inputData[0] = 1;
                    else
                        inputData[64 * 7 + 1] = 1;
                    continue;
                }

                if (spaceCount == 2)
                {
                    if (fen[i] == '-')
                        continue;

                    int multiplier = 0;
                    if (char.IsLower(fen[i]))
                        multiplier = 1;

                    if (char.ToLower(fen[i]) == 'k')
                        inputData[1 + (64 * 7 + 2) * multiplier] = 1;
                    else if (char.ToLower(fen[i]) == 'q')
                        inputData[2 + (64 * 7 + 1) * multiplier] = 1;
                }

                if (spaceCount == 3)
                {
                    if (fen[i] == '-')
                        continue;

                    if (char.IsDigit(fen[i]))
                    {
                        row = int.Parse(fen[i].ToString()) - 1;
                        inputData[row * column * 7 * ((int)inputData[64 * 7 + 1] + 1) + 7] = 1;
                    }
                    else
                    {
                        column = (fen[i] % 32) - 1;
                    }
                }
                if (spaceCount == 4)
                {
                    inputData[64 * 7 * 2 + 2 + 4] = int.Parse(fen[i].ToString());
                    break;
                }
            }
            return inputData;
        }*/

        static void Square()
        {
            string fileName = "Squares.txt";
            NeuralNetwork nn = new NeuralNetwork(1, hyperParameters, CostFunction.CostType.MSE);
            nn.AddLayers(Activation.ActivationType.Sigmoid, 8, 8);
            nn.AddLayers(Activation.ActivationType.Linear, 1);

            Random rng = new Random();

            List<InputData> learnDatas = new List<InputData>();
            nn.LoadWeights(path + fileName);

            /*for (double x = 0; x <= 1; x += 0.0001d)
            {
                if (x % 0.01 == 0)
                    Console.Write("--");
                var res = nn.Calculate(new double[] { x });
                Console.WriteLine(Math.Round(res[0], 4) + ", expected: " + Math.Round((x * x * x + 12 * x + 5),4));
            }*/

            for (int i = -100; i < 100; i++)
            {
                double x = (i * 0.01);
                learnDatas.Add(new InputData(new double[] { x }, new double[] { x * x * x + 12 * x + 5 }));
            }
            var l = learnDatas.ToArray();
            for (int i = 0; i < 100000; i++)
            {
                nn.Learn(l, currentLearnRate, 4);
                if (i % 10 == 0)
                {
                    l = Shuffle(rng, l);
                }
                EpochCompleted(i);
            }
            nn.SaveWeights(path + fileName);
        }

        public static T[] GetArrayFromIndex<T>(T[] array, int index, int amount)
        {
            T[] result = new T[amount];

            for (int i = index; i < index + amount; i++)
            {
                result[i-index] = array[i];
            }
            return result;
        }

        public static T[] Shuffle<T>(Random rng, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
            return array;
        }
        static void EpochCompleted(int epochCount)
        {
            currentLearnRate = (1.0 / (1.0 + hyperParameters.learnRateDecay * epochCount)) * hyperParameters.initialLearningRate;
        }

        static void AND()
        {
            NeuralNetwork nn = new NeuralNetwork(2, hyperParameters);
            nn.AddLayers(Activation.ActivationType.Linear, 2);
            nn.AddLayers(Activation.ActivationType.Linear, 1);

            Random rand = new Random();

            for (int i = 0; i < 3000; i++)
            {
                int x = (int)Math.Round(rand.NextDouble());
                int y = (int)Math.Round(rand.NextDouble());
                var input = new InputData(new double[] { x, y }, new double[] { x & y });

                nn.Learn(input, 0.1);
            }
        }

        static void XOR()
        {
            hyperParameters = new HyperParameters();
            NeuralNetwork nn = new NeuralNetwork(2, hyperParameters, CostFunction.CostType.MSLE);
            nn.AddLayers(Activation.ActivationType.ReLU, 2);
            nn.AddLayers(Activation.ActivationType.Linear, 1);

            Random rand = new Random();

            for (int i = 0; i < 30000000; i++)
            {
                int x = (int)Math.Round(rand.NextDouble());
                int y = (int)Math.Round(rand.NextDouble());
                var input = new InputData(new double[] { x, y }, new double[] { x ^ y });

                nn.Learn(input, 0.01);
            }
        }
    }
}
