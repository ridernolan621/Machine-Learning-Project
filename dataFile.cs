using Microsoft.ML.Data;

namespace project2{
    public class Data{
    
        [LoadColumn(0)]
        public string? Date { get; set; }

        [LoadColumn(1)]
        public float Open { get; set; }

        [LoadColumn(2)]
        public float High { get; set; }

        [LoadColumn(3)]
        public float Low { get; set; }

        [LoadColumn(4)]
        public float Close { get; set; }

        [LoadColumn(5)]
        public float PipChange { get; set; }

        [LoadColumn(6)]
        public float PercentChange { get; set; }
    }   

    public class NewData{
        [LoadColumn(0)]
        public string? Date { get; set; }

        [LoadColumn(1)]
        public float Open { get; set; }

        [LoadColumn(2)]
        public float High { get; set; }

        [LoadColumn(3)]
        public float Low { get; set; }

        [LoadColumn(4)]
        public float Close { get; set; }

        //[LoadColumn(5)]
        //public float Volume { get; set; }
    }

    public class Prediction{
        [ColumnName("Score")]
        public float PredictedClose { get; set; }
    }

    public class NewPrediction{
        [ColumnName("Score")]
        public float PredictedClose { get; set; }
    }
}