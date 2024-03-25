import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{SparkSession, Row}

val spark = SparkSession.builder.appName("TFIDFClustering").getOrCreate()

// 读取HDFS上的文本文件
val data = spark.read.text("hdfs:/arxiv_data/processed_abstracts.txt").toDF("text")

// Tokenizer用于将文本列分解成单词列表
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
val wordsData = tokenizer.transform(data)

// 使用HashingTF将单词转换为原始特征向量
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100000)
val featurizedData = hashingTF.transform(wordsData)

// 使用IDF转换TF特征
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)
val tfidfData = idfModel.transform(featurizedData)

// 将TF-IDF特征转换为稠密向量
val denseVectorData = tfidfData.select("features").rdd.map {
  case Row(features: Vector) => (features.toDense, features)
}.toDF("denseFeatures", "features")

import org.apache.spark.sql.functions._
// denseVectorData 是DataFrame，包含了一个名为 'denseFeatures' 的列
// 将结构化的特征转换为字符串
val stringData = denseVectorData.withColumn("denseFeaturesString", col("denseFeatures").cast("string"))

// 保存为文本文件
stringData.select("denseFeaturesString").write.text("hdfs:/arxiv_data/TF_IDF_Result.txt")

// 设置聚类的数量
val k = 8

// 创建K-Means实例并设置参数
val kmeans = new KMeans().setK(k).setSeed(1L)

// 训练模型
val model = kmeans.fit(denseVectorData)

// 获取数据的聚类分配
val predictions = model.transform(denseVectorData)

import org.apache.spark.sql.functions.col

// predictions 是 DataFrame，包含了一个名为 'prediction' 的列
// 将整数类型的 'prediction' 列转换为字符串
val stringPredictions = predictions.withColumn("predictionString", col("prediction").cast("string"))

// 保存为文本文件
stringPredictions.select("predictionString").write.text("hdfs:/arxiv_data/Kmeans_Result.txt")


spark.stop()
