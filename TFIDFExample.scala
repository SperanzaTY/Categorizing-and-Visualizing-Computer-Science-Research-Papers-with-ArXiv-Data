import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row,SparkSession}

val spark = SparkSession.builder.appName("TFIDFClustering").getOrCreate()

// Read the CSV file
val data = spark.read.option("header", "false").csv("hdfs:/arxiv_data/abstracts.csv")

// Assuming the column name in the CSV is "_c0" (default column name for CSV without header)
val textColumn = "_c0"

// Tokenize the text column
val tokenizer = new Tokenizer().setInputCol(textColumn).setOutputCol("words")
val wordsData = tokenizer.transform(data)

// Apply TF (term frequency) on the tokenized words
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10000)
val featurizedData = hashingTF.transform(wordsData)

// Apply IDF (inverse document frequency) on the TF features
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)
val tfidfData = idfModel.transform(featurizedData)

// Save the TF-IDF vectors to a file
//val outputData=tfidfData.select("features").rdd.map{case Row(features)=>features}
//outputData.saveAsTextFile("hdfs:/arxiv_data/jieguo")

// Convert the TF-IDF features to dense vectors
val denseVectorData = tfidfData.select("features").rdd.map {case Row(features: Vector) => (features.toDense, features)}.toDF("denseFeatures", "features")

// Set the number of clusters
val k = 8

// Create a K-means instance
val kmeans = new KMeans().setK(k).setSeed(1L)

// Fit the model to the data
val model = kmeans.fit(denseVectorData)

// Get the cluster assignments for the data
val predictions = model.transform(denseVectorData)

// Show the cluster assignments
predictions.select("denseFeatures", "prediction").show()

// Save the cluster assignments to a file
val output=predictions.select("prediction").rdd.map{case Row(prediction)=>prediction}
output.saveAsTextFile("hdfs:/arxiv_data/finaljieguo")

spark.stop()
