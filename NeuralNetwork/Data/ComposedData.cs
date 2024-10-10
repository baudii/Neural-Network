using System;
using System.Diagnostics;

namespace KKNeuralNetwork.Data
{
	/// <summary>
	/// Class is used to get a learning state of the network
	/// </summary>
	public class ComposedData
	{
		Stopwatch stopwatch;

		double[] predictedOutputs, expectedOutputs, initialInputs;

		double SummedCost;
		double ElapsedIterations;

		double averageCost;

		double logUpdateTime;

		/// <summary>
		/// Activates automatic logging every N seconds
		/// </summary>
		/// <param name="updateTime">Delay between logs in seconds</param>
		public void ActivateLog(double updateTime)
		{
			logUpdateTime = updateTime;
		}

		/// <summary>
		/// Returns the average cost of your neural network since the last time you accessed this field.
		/// If no iterations elapsed returns previously calculated average cost
		/// You can use this to get some sort of graph and see how network trains realtime
		/// </summary>
		public double AverageCost
		{
			get
			{
				if (ElapsedIterations > 0)
				{
					averageCost = SummedCost / ElapsedIterations;
					Reset();
				}

				return averageCost;
			}
		}

		internal ComposedData()
		{
			stopwatch = new Stopwatch();
			stopwatch.Start();
			initialInputs = new double[0];
			expectedOutputs = new double[0];
			predictedOutputs = new double[0];
		}

		void Reset()
		{
			stopwatch.Reset();
			SummedCost = 0;
			ElapsedIterations = 0;
		}

		internal void UpdateData(ref double iterationCost, ref double[] initialInputs, ref double[] expectedOutputs, ref double[] predictedOutputs)
		{
			lock (stopwatch)
			{
				SummedCost += iterationCost;
				ElapsedIterations++;
				this.initialInputs = initialInputs;
				this.expectedOutputs = expectedOutputs;
				this.predictedOutputs = predictedOutputs;

				if (logUpdateTime > 0 && stopwatch.Elapsed.TotalSeconds > logUpdateTime)
				{
					LogIterationData();
					stopwatch.Restart();
				}
			}
		}

		/// <summary>
		/// Logs out a learning state of the network. Used for debugging and monitoring the learning process
		/// </summary>
		public void LogIterationData()
		{
			int pad = 20;
			PrintCenteredText(" Neural Network Training Log ", '-', pad * 3);
			Console.WriteLine($"{"Average Cost:".PadRight(pad)}{AverageCost}");

			Console.Write("Initial Inputs:".PadRight(pad));
			OutputArray(initialInputs);
			Console.Write("Expected Outputs:".PadRight(pad));
			OutputArray(expectedOutputs);
			Console.Write("Predicted Outputs:".PadRight(pad));
			OutputArray(predictedOutputs);

			Console.WriteLine("".PadRight(pad * 3, '-'));
			Console.WriteLine();
		}

		static void OutputArray(double[] array)
		{
			if (array.Length == 0)
			{
				Console.WriteLine("[]");
				return;
			}

			Console.Write("[" + array[0]);
			for (int i = 1; i < array.Length; i++)
				Console.Write(", " + array[i].ToString());
			Console.WriteLine("]");
		}
		static void PrintCenteredText(string text, char fillChar, int totalWidth)
		{
			// Добавляем половину символов заполнения слева и справа
			int padding = (totalWidth - text.Length) / 2;
			string result = text.PadLeft(padding + text.Length, fillChar).PadRight(totalWidth, fillChar);
			Console.WriteLine(result);
		}
	}
}
