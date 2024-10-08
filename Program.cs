using System.CodeDom;
using Microsoft.ML;

namespace project2{
    class Program {
        static void Main(string[] args){
            Console.Clear();

            // Create new ML context
            var context = new MLContext();
            var dataPath = "EURUSD_historical_data.csv";
            var modelPath = "model.zip"; 

            var newModelPath = "retrainedModel.zip";
            var newDataPath = "EURUSD1440.csv";

            // Check if the model already exists
            ITransformer? trainedModel = null;

            if (File.Exists(modelPath)){

                // Load the existing model
                trainedModel = context.Model.Load(modelPath, out var modelInputSchema);

                Console.WriteLine("Model loaded successfully.");
            }else{

                // Load data from text file into IDataView
                IDataView dataView = context.Data.LoadFromTextFile<Data>(
                    dataPath,
                    separatorChar: ',',
                    hasHeader: true
                );

                // Define the learning pipeline
                var firstPipeline = context.Transforms.Concatenate("Features", "Open", "High", "Low")
                    .Append(context.Regression.Trainers.FastTree(labelColumnName: "Close"));

                // Save the trained model
                try{
                    // Train the model
                    trainedModel = firstPipeline.Fit(dataView);
                    Console.WriteLine("Model trained successfully.");

                    context.Model.Save(trainedModel, dataView.Schema, modelPath);
                    Console.WriteLine($"Model saved to {modelPath}");
                }catch (Exception ex){
                    Console.WriteLine($"Error: {ex.Message}");

                    Environment.Exit(0);
                }
            }

            // Call the retrain function with the path to your model and new training data
            ITransformer retrainedModel;
            
            retrainedModel = RetrainModel(modelPath, newDataPath, newModelPath);
 
            // Create prediction engine (For single predictions)
            var predEngine = context.Model.CreatePredictionEngine<NewData, NewPrediction>(retrainedModel);

            // Prepare sample input data
            var sampleData = new NewData(){
                Open = 1.09758f,
                High = 1.09810f,
                Low = 1.09753f,
            };

            // Predict the closing price
            var result = predEngine.Predict(sampleData);

            // Output the predicted closing price
            Console.WriteLine($"Predicted Closing Price: {result.PredictedClose}");
        }











        private static ITransformer RetrainModel(string modelPath, string newDataPath, string newModelPath){

            var context = new MLContext();

            ITransformer existingModel = context.Model.Load(modelPath, out var modelInputSchema);

            if (File.Exists(modelPath)){

                Console.WriteLine("Existing model loaded successfully. Starting retraining");

                IDataView dataView = context.Data.LoadFromTextFile<NewData>(
                    newDataPath,
                    separatorChar: '\t',
                    hasHeader: false
                );

                Console.WriteLine("New training data loaded successfully.");

                var pipeline = context.Transforms.Concatenate("Features", "Open", "High", "Low")
                    .Append(context.Regression.Trainers.FastTree(labelColumnName: "Close"));

                try{

                    ITransformer retrainedModel = pipeline.Fit(dataView);

                    Console.WriteLine("Model retrained successfully.");

                    context.Model.Save(retrainedModel, dataView.Schema, newModelPath);
                    Console.WriteLine($"Retrained model saved to {newModelPath}");

                    return retrainedModel;

                }catch(Exception ex){
                    Console.WriteLine($"Error during retraining: {ex.Message}");
                    
                    return existingModel;
                }
            }else{
                Console.WriteLine("Error");
                return existingModel;
            }
        }
    }
}