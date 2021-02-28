using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        private static readonly bool _useTrainedData = false;

        static void Main(string[] args)
        {
            double tick = DateTime.Now.Second + DateTime.Now.Millisecond / 1000, tock;
            MLContext mlContext = new MLContext();

            if (_useTrainedData)
            {
                DataViewSchema modelSchema;
                ITransformer trainedModel = mlContext.Model.Load("data.zip", out modelSchema);
                UseModelWithBatchItems(mlContext, trainedModel);
                tock = DateTime.Now.Second + DateTime.Now.Millisecond / 1000;
                Console.WriteLine($"tick{tick}:{tock}\n{(tock - tick) * 1000}");
                return;
            }

            TrainTestData splitDataView = LoadData(mlContext, out IDataView dataview);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            mlContext.Model.Save(model, dataview.Schema, "data.zip");
            Evaluate(mlContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(mlContext, model);
            UseModelWithBatchItems(mlContext, model);
            tock = DateTime.Now.Second + DateTime.Now.Millisecond / 1000;
            Console.WriteLine($"tick{tick}:{tock}\n{(tock - tick) * 1000}");
        }
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {

            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                // SentimentText = "This was a very bad steak"
                SentimentText = "A carne estava ruim"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {

            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                    // SentimentText = "Esta foi uma refeição horrível"
                },
                new SentimentData
                {
                    // SentimentText = "Eu amo esse espaguete."
                    SentimentText = "I love this spaghetti."
                },
                new SentimentData
                {
                    // SentimentText = "A pizza estava incrível."
                    SentimentText = "The pizza was amazing."
                },
                new SentimentData
                {
                    // SentimentText = "Não vou comer aqui de novo."
                    SentimentText = "I will not eat here again."
                }
            };
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
        public static TrainTestData LoadData(MLContext mlContext, out IDataView dataView)
        {
            dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.7);

            return splitDataView;

        }
    }

}
