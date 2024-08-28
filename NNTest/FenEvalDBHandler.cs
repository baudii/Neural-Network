using System.Data.SQLite;

namespace NNTest
{
    public static class FenEvalDBHandler
    {
        const string db_path = @"URI=file:D:\Загрузки\\LichessDB\\test.db";
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

        public class FenEval
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

