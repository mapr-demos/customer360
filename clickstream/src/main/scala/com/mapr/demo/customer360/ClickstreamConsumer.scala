package com.mapr.demo.customer360
/** ****************************************************************************
  * PURPOSE:
  *
  * Consume clickstream data from a prepopulated topic in MapR Streams.
  *
  * USAGE:
  *
  * /opt/mapr/spark/spark-2.1.0/bin/spark-submit --class com.mapr.demo.customer360.ClickstreamConsumer --master local[2] target/mapr-sparkml-streaming-uber-1.0.jar /tmp/clickstream:401k
  *
  * AUTHOR:
  * Ian Downard, idownard@mapr.com
  *
  * ****************************************************************************/
import org.apache.kafka.clients.consumer.ConsumerConfig

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._

import org.apache.spark.streaming.{ Seconds, StreamingContext, Time }
import org.apache.spark.streaming.kafka09.{ ConsumerStrategies, KafkaUtils, LocationStrategies }
import org.apache.spark.streaming.kafka.producer._

object ClickstreamConsumer {
  case class UberC(dt: String, lat: Double, lon: Double, base: String, cluster: String) extends Serializable
  def main(args: Array[String]) = {
    if (args.length < 1) {
      System.err.println("Usage: SparkKafkaConsumer <topic consume> ")
      System.exit(1)
    }

    val schema = StructType(Array(
      StructField("dt", TimestampType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("base", StringType, true),
      StructField("cluster", StringType, true)
    ))
    val groupId = "testgroup"
    val offsetReset = "earliest"
    val pollTimeout = "5000"
    val Array(topicc) = args
    val brokers = "maprdemo:9092" // not needed for MapR Streams, needed for Kafka

    val sparkConf = new SparkConf()
      .setAppName(ClickstreamConsumer.getClass.getName)

    val ssc = new StreamingContext(sparkConf, Seconds(2))

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

    valuesDStream.foreachRDD { (rdd: RDD[String], time: Time) =>
      // There exists at least one element in RDD
      if (!rdd.isEmpty) {
        val count = rdd.count
        println("count received " + count)
        val spark = SparkSession.builder.config(rdd.sparkContext.getConf).getOrCreate()
        import spark.implicits._

        import org.apache.spark.sql.functions._
        val df: Dataset[UberC] = spark.read.schema(schema).json(rdd).as[UberC]
        df.show
        df.createOrReplaceTempView("uber")

        df.groupBy("cluster").count().show()

        spark.sql("select cluster, count(cluster) as count from uber group by cluster").show

        spark.sql("SELECT hour(uber.dt) as hr,count(cluster) as ct FROM uber group By hour(uber.dt)").show

        df.groupBy("cluster").count().show()

        val countsDF = df.groupBy($"cluster", window($"dt", "1 hour")).count()
        countsDF.createOrReplaceTempView("uber_counts")

        spark.sql("select cluster, sum(count) as total_count from uber_counts group by cluster").show
        //spark.sql("sql select cluster, date_format(window.end, "MMM-dd HH:mm") as dt, count from uber_counts order by dt, cluster").show

        spark.sql("select cluster, count(cluster) as count from uber group by cluster").show

        spark.sql("SELECT hour(uber.dt) as hr,count(cluster) as ct FROM uber group By hour(uber.dt)").show
      }
    }

    ssc.start()
    ssc.awaitTermination()

    ssc.stop(stopSparkContext = true, stopGracefully = true)
  }

}

