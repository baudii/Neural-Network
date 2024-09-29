using System.Data.SQLite;

namespace NNTest
{
	/// <summary>
	/// A static class responsible for handling database operations related to retrieving
	/// chess board FEN strings and their corresponding evaluation scores from a SQLite database.
    /// Designed specifically for neural network training
	/// </summary>
	public static class FenEvalDBHandler
    {
		const string db_path = @"URI=file:D:\Загрузки\\LichessDB\\test.db";


		/// <summary>
		/// Retrieves a specified number of FEN strings and their evaluations from the database.
		/// The FEN strings are fetched based on their ID range.
		/// </summary>
		/// <param name="from">The starting ID for the database query (inclusive).</param>
		/// <param name="amount">The number of records to retrieve.</param>
		/// <returns>An array of FenEval objects containing the FEN strings and corresponding evaluations.</returns>
		public static FenEval[] GetFenStrings(int from, int amount)
        {
            using var connection = new SQLiteConnection(db_path);
            connection.Open();
            string command = $"SELECT fen, eval FROM evaluations WHERE id BETWEEN {from} AND {from + amount - 1};";

            using var cmd = new SQLiteCommand(command, connection);
            using SQLiteDataReader reader = cmd.ExecuteReader();

            FenEval[] fenEvals = new FenEval[amount];
            int i = 0;
            while (reader.Read())
            {
                fenEvals[i] = new FenEval(reader.GetString(0), reader.GetFloat(1));
                i++;
            }

            return fenEvals;
        }

		public static double[] DecipherFen(string fen)
		{
			// FEN example:
			// rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

			int cell = 0;
			double[] inputData = new double[64 * 2 + 6]; // (cells * sides + whose turn is it + castles  + 50moveCounter
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

		/// <summary>
		/// A struct that represents a combination of a FEN string and its corresponding evaluation score.
		/// </summary>
		public struct FenEval
        {
            public string fen;
            public float eval;
            public FenEval(string fen, float eval)
            {
                this.fen = fen;
                this.eval = eval;
            }
        }
    }
}

