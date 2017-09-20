package com.mapr.demo.customer360
/** ****************************************************************************
  * PURPOSE:
  *
  * Demonstrate how to load an RDD from MapR Streams, join it with an RDD loaded from a table in MapR-DB, perform some ML on that joined RDD with Spark ML, and return the ML result back to the master data table in MapR-DB. Specifically, this example will consume clickstream data from /tmp/clickstream:weblog, join it with the crm_table in MapR-DB, perform some rudimentary Spark ML code on that combined data set to predict churn risk, then persist the result back to MapR-DB.
  *
  * USAGE:
  *
  * /opt/mapr/spark/spark-2.1.0/bin/spark-submit --class com.mapr.demo.customer360.ChurnRiskAssessor --master local[2] target/mapr-sparkml-streaming-customer360-1.0.jar
  *
  * AUTHOR:
  * Ian Downard, idownard@mapr.com
  *
  * ****************************************************************************/
import org.apache.kafka.clients.consumer.ConsumerConfig

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.streaming.{ Seconds, StreamingContext, Time }
import org.apache.spark.streaming.kafka09.{ ConsumerStrategies, KafkaUtils, LocationStrategies }
import org.apache.spark.streaming.kafka.producer._

object ChurnRiskAssessor {

  case class Click(user_id: Integer, datetime: String, os: String, browser: String, response_time_ms: String, product: String, url: String) extends Serializable
  def main(args: Array[String]) = {

    val schema = StructType(Array(
      StructField("user_id", IntegerType, true),
      StructField("datetime", StringType, true),
      StructField("os", StringType, true),
      StructField("browser", StringType, true),
      StructField("response_time_ms", StringType, true),
      StructField("product", StringType, true),
      StructField("url", StringType, true)
    ))

    val groupId = "clickstream_reader"
    val offsetReset = "earliest"
    val pollTimeout = "5000"
    val Array(topicc) = Array("/tmp/clickstream:weblog")
    val brokers = "kafkabroker.example.com:9092" // not needed for MapR Streams, needed for Kafka

    val sparkConf = new SparkConf().setAppName("Spark demo")

    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val ssc = new StreamingContext(sc, Seconds(2))

    val topicsSet = topicc.split(",").toSet

    val kafkaParams = Map[String, String](
      ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG -> brokers,
      ConsumerConfig.GROUP_ID_CONFIG -> groupId,
      ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG ->
        "org.apache.kafka.common.serialization.StringDeserializer",
      ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG ->
        "org.apache.kafka.common.serialization.StringDeserializer",
      ConsumerConfig.AUTO_OFFSET_RESET_CONFIG -> offsetReset,
      ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG -> "false",
      "spark.kafka.poll.time" -> pollTimeout
    )

    val consumerStrategy = ConsumerStrategies.Subscribe[String, String](topicsSet, kafkaParams)
    val messagesDStream = KafkaUtils.createDirectStream[String, String](
      ssc, LocationStrategies.PreferConsistent, consumerStrategy
    )

    val valuesDStream = messagesDStream.map(_.value())
    import org.apache.spark.sql.functions._
    val spark = SparkSession
      .builder()
      .appName("Spark demo")
      .config(sparkConf)
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames

    valuesDStream.foreachRDD { (rdd: RDD[String], time: Time) =>
      // There exists at least one element in RDD
      if (!rdd.isEmpty) {
        val count = rdd.count
        println("count received " + count)

        val df: Dataset[Click] = spark.read.schema(schema).json(rdd).as[Click]
        df.show
        df.createOrReplaceTempView("weblog_snapshot")
        spark.sql("select count(*) from weblog_snapshot").show
      }
    }

    ssc.start()
    ssc.awaitTerminationOrTimeout(10 * 1000)
    ssc.stop(stopSparkContext = false, stopGracefully = false)


    // Load data from MapR-DB
    import com.fasterxml.jackson.annotation.{JsonIgnoreProperties, JsonProperty}
    import org.apache.spark.{SparkConf, SparkContext}
    import com.mapr.db.spark._
    import com.mapr.db.spark.impl.OJAIDocument

    val rdd = sc.loadFromMapRDB("/mapr/my.cluster.com/tmp/crm_data").select("name", "address", "first_visit", "zip", "sentiment", "persona", "churn_risk");
    println("Number of records loaded from MapR-DB: " + rdd.count)
    val stringrdd = rdd.map(x => x.getDoc.asJsonString())

    val crm_df = sqlContext.read.json(stringrdd)
    // add a user_id to this dataframe to make it easier to join with the clickstream
    crm_df.withColumn("user_id",monotonicallyIncreasingId).createOrReplaceTempView("crm_table")
    crm_df.show(5)

    spark.sql("select * from weblog_snapshot limit 2").show
    spark.sql("select * from crm_table limit 2").show

    //  Combine RDDs from MapR-DB and MapR-Streams for Spark M

    val joinedDF = spark.sql("SELECT weblog_snapshot.datetime, weblog_snapshot.os, weblog_snapshot.browser, weblog_snapshot.response_time_ms,weblog_snapshot.product,weblog_snapshot.url, crm_table.*, case when crm_table.churn_risk >= 20 then 1 else 0 end as churn_label from weblog_snapshot JOIN crm_table ON weblog_snapshot.user_id == crm_table.user_id")
    joinedDF.count()
    joinedDF.show(2)

    // Predict Churn with Spark ML

    import org.apache.spark._
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types._
    import org.apache.spark.sql._
    import org.apache.spark.sql.Dataset
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
    import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
    import org.apache.spark.ml.feature.StringIndexer
    import org.apache.spark.ml.tuning.ParamGridBuilder
    import org.apache.spark.ml.tuning.CrossValidator
    import org.apache.spark.ml.tuning.CrossValidatorModel
    import org.apache.spark.ml.feature.VectorAssembler

    val sameCVModel = CrossValidatorModel.load("/tmp/churn_prediction_model")
    val predictions_df = sameCVModel.transform(joinedDF).select("_id", "user_id", "prediction")
    predictions_df.show(2)
    predictions_df.groupBy("prediction").count.show

    predictions_df.write.mode(SaveMode.Overwrite).format("json").save("predictions.json")

    val filesRdd = sc.wholeTextFiles("maprfs:/user/mapr/predictions.json")
    val maprd = filesRdd.map(file => MapRDBSpark.newDocument(file._2))
//    val maprd = rdd2.map(str => MapRDBSpark.newDocument(str))
    maprd.saveToMapRDB("/tmp/realtime_churn_predictions", createTable = false, bulkInsert = true, idFieldPath = "id")

  }
}
