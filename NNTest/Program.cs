using System.Diagnostics;

namespace NNTest
{
    public class Program
    {
        static Dictionary<char, int> pieceTypes = new Dictionary<char, int>() { { 'p', 0 }, { 'n', 1 }, { 'b', 2 }, { 'r', 3 }, { 'q', 4 }, { 'k', 5 }, };
        static double currentLearnRate;
        static HyperParameters hyperParameters = new HyperParameters();

        public static string path = "C:\\Users\\Хожик\\Desktop\\Weights\\";


        static void Main(string[] args)
        {
            //ChessTrainer();
            ChessTest(2700);
            //XOR();
            //XOR_Test();
            //Square();
            //SquareTest();
        }

        static void ChessTest(int startFrom)
        {
            Random rand = new Random();
            var nn = new NeuralNetwork(64 * 2 + 6, hyperParameters, CostFunction.CostType.MSE);
            nn.AddLayers(Activation.ActivationType.TanH, 128, 64, 32, 16);
            nn.AddLayers(Activation.ActivationType.Linear, 1);
            for (int i = 0; i < 64; i++)
            {
                var chessPosition = FenEvalDBHandler.GetFenStrings(startFrom + i, 1);

                var dfen = DecipherFen(chessPosition[0].fen);

                var result = nn.Calculate(dfen);

                Console.WriteLine("Expected: " + (chessPosition[0].eval) + "\nOutput: " + result[0] * 40);
            }
        }

        static void ChessTrainer()
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();


            hyperParameters = new HyperParameters();
            var fileName = "ChessWeights_4.txt";
            Random rand = new Random();
            currentLearnRate = hyperParameters.initialLearnRate;
            var nn = new NeuralNetwork(64 * 2 + 6, hyperParameters, CostFunction.CostType.MSE);
            nn.AddLayers(Activation.ActivationType.TanH, 128, 64, 32, 16);
            nn.AddLayers(Activation.ActivationType.Linear, 1);

            if (!nn.LoadWeights(path + fileName))
                Console.WriteLine("Weights were not loaded");

            var databaseArray = FenEvalDBHandler.GetFenStrings(2600, 400);
            TrainingData[] trainingData = new TrainingData[databaseArray.Length];
            int b = 0;
            foreach (var item in databaseArray)
            {
                var desipheredFen = DecipherFen(item.fen);

                trainingData[b] = new TrainingData(desipheredFen, new double[] { item.eval / 40 });
                b++;
            }
            for (int epoch = 0; epoch < 100; epoch++)
            {
                for (int i = 0; i < 50; i++)
                {
                    nn.Learn(trainingData, 0.01d, 32);
                }

                trainingData = Shuffle(rand, trainingData);
                EpochCompleted(epoch);

                if (stopwatch.Elapsed.TotalSeconds > 5)
                {
                    stopwatch.Restart();
                    nn.SaveWeights(path + fileName);
                    Console.WriteLine("------------------ SAVED ---------------");
                    Console.WriteLine(epoch + "%");
                }
            }
            nn.SaveWeights(path + fileName);

            Console.WriteLine("------------------ FINISHED ---------------");
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
            // FEN example:
            //rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

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
                else if (spaceCount == 1)
                {
                    if (fen[i] == 'w')
                        inputData[128] = 1;
                    else
                        inputData[128] = -1;
                }
                else if (spaceCount == 2)
                {
                    if (fen[i] == '-')
                        continue;

                    int castleIndex = 129;
                    if (char.IsLower(fen[i]))
                        castleIndex = 131;

                    if (char.ToLower(fen[i]) == 'k')
                        inputData[castleIndex] = 1;
                    else if (char.ToLower(fen[i]) == 'q')
                        inputData[castleIndex + 1] = 1;
                }
                else if (spaceCount == 3)
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
                else if (spaceCount == 4)
                {
                    inputData[133] = int.Parse(fen[i].ToString());
                    break;
                }
            }
            return inputData;
        }
/*        static double[] DecipherFen(string fen)
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

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            string fileName = "Squares_Relu_8_8_4.txt";
            NeuralNetwork nn = new NeuralNetwork(1, hyperParameters, CostFunction.CostType.MSE);
            nn.AddLayers(Activation.ActivationType.ReLU, 8, 8, 4);
            nn.AddLayers(Activation.ActivationType.Linear, 1);
            currentLearnRate = 0.01f; 

            Random rng = new Random();

            List<TrainingData> learnDatas = new List<TrainingData>();
            nn.LoadWeights(path + fileName);

            /*for (double x = 0; x <= 1; x += 0.0001d)
            {
                if (x % 0.01 == 0)
                    Console.Write("--");
                var res = nn.Calculate(new double[] { x });
                Console.WriteLine(Math.Round(res[0], 4) + ", expected: " + Math.Round((x * x * x + 12 * x + 5),4));
            }*/

            for (double x = 0; x < 100; x++)
            {
                var normalizedInput = x * 0.01d;
                learnDatas.Add(new TrainingData(new double[] { normalizedInput }, new double[] { x * x * 0.0001 }));
            }
            var l = learnDatas.ToArray();

            int i = 0;
            while (true)
            {
                nn.Learn(l, 0.01f, 8); ;
                if (i % 10 == 0)
                {
                    l = Shuffle(rng, l);
                }
                if (stopwatch.Elapsed.TotalSeconds > 10)
                {
                    stopwatch.Restart();
                    nn.SaveWeights(path + fileName);
                    Console.WriteLine("------------------ SAVED ---------------");
                }
                EpochCompleted(i);
                i++;
            }
        }

        static void SquareTest()
        {
            NeuralNetwork nn = new NeuralNetwork(1, hyperParameters, CostFunction.CostType.MSE);
            string fileName = "Squares.txt";
            nn.AddLayers(Activation.ActivationType.Sigmoid, 8, 8);
            /*            string fileName = "Squares_Relu_4_4_1.txt";
                        nn.AddLayers(Activation.ActivationType.ReLU, 4, 4);*/

            nn.AddLayers(Activation.ActivationType.Linear, 1);
            nn.LoadWeights(path + fileName);



            Console.WriteLine("X= (0<X<100)");
            string xstr = Console.ReadLine();
            double.TryParse(xstr, out var x);
            Console.WriteLine("You entered: " + x);
            Console.WriteLine("X^2=");
            double[] result = nn.Calculate(new double[] { x * 0.01d });
            Console.WriteLine(result[0] * 10000);
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
            currentLearnRate = hyperParameters.initialLearnRate / (1.0 + hyperParameters.learnRateDecay * epochCount);
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
                var input = new TrainingData(new double[] { x, y }, new double[] { x & y });

                nn.Learn(input, 0.1);
            }
        }

        static void XOR()
        {
            Stopwatch sw = Stopwatch.StartNew();
            string fileName = "XOR_Relu_2_2_1_MSLE.txt";
            hyperParameters = new HyperParameters();
            NeuralNetwork nn = new NeuralNetwork(2, hyperParameters, CostFunction.CostType.MSLE);
            nn.AddLayers(Activation.ActivationType.ReLU, 2);
            nn.AddLayers(Activation.ActivationType.Linear, 1);

            currentLearnRate = hyperParameters.initialLearnRate;
            nn.LoadWeights(path + fileName);
            Random rand = new Random();

            for (int i = 0; i < 30000000; i++)
            {
                int x = (int)Math.Round(rand.NextDouble());
                int y = (int)Math.Round(rand.NextDouble());
                var input = new TrainingData(new double[] { x, y }, new double[] { x ^ y });

                nn.Learn(input, 0.0001);

                
            }

            nn.SaveWeights(path + fileName);
        }

        static void XOR_Test()
        {
            string fileName = "XOR_Relu_2_2_1_MSLE.txt";
            hyperParameters = new HyperParameters();
            NeuralNetwork nn = new NeuralNetwork(2, hyperParameters, CostFunction.CostType.MSLE);
            nn.AddLayers(Activation.ActivationType.ReLU, 2);
            nn.AddLayers(Activation.ActivationType.Linear, 1);

            nn.LoadWeights(path + fileName);

            Console.WriteLine("A=");
            string astr = Console.ReadLine();
            int.TryParse(astr, out int a);
            Console.WriteLine("B=");
            string bstr = Console.ReadLine();
            int.TryParse(bstr, out int b);
            Console.WriteLine("A^B=" + (a^b));
            Console.WriteLine("NN prediction:");
            double[] result = nn.Calculate(new double[] { a, b });
            Console.WriteLine(result[0]);

        }
    }
}
