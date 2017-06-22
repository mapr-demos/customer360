package com.mapr.demo.customer360

/******************************************************************************
  * PURPOSE:
  *
  * Monitor a specified directory and load the files created in that directory
  * to a MapR-DB table. Assumes all the files contain only valid JSON.
  *
  * PREREQUISITES:
  *
  * MapR release: 5.2.1 or later
  * MEP 3.0 or later
  *
  * EXAMPLE USAGE:
  *
  * /opt/mapr/spark/spark-2.1.0/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar /mapr/demo.mapr.com/ingest/ /tmp/mytable
  *
  * /opt/mapr/spark/spark-2.1.0/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar /mapr/demo.mapr.com/data.json /tmp/mytable
  *
  * REFERENCES:
  *
  * MapR-DB / Spark RDD connector:
  *   http://maprdocs.mapr.com/home/Spark/NativeSparkConnectorJSON.html
  *
  * Spark fileStream:
  *   http://spark.apache.org/docs/latest/streaming-programming-guide.html#basic-sources
  *
  *****************************************************************************/

import com.mapr.db.MapRDB
import com.mapr.db.spark.MapRDBSpark
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import java.io.File

object LoadJsonToMaprDB {

  def main(args: Array[String]) = {
    if (args.length < 1) {
      System.err.println("Usage: LoadJsonToMaprDB <json file or directory> <mapr-db table> ")
      System.err.println("Example: /opt/mapr/spark/spark-2.1.0/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar /mapr/demo.mapr.com/ingest/ /tmp/mytable")
      System.err.println("Example: /opt/mapr/spark/spark-2.1.0/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar /mapr/demo.mapr.com/data.json /tmp/mytable")
      System.exit(1)
    }
    var Array(dataDirectory, tableName) = args

    if (!MapRDB.tableExists(tableName)) {
      System.out.println("creating table " + tableName)
      MapRDB.createTable(tableName)
    }

    val sparkConf = new SparkConf()
      .setAppName(LoadJsonToMaprDB.getClass.getName)

    val sc = new SparkContext(sparkConf)
    val ssc = new StreamingContext(sc, Seconds(2))


    val d = new File(dataDirectory)
    if (!d.exists) {
      System.err.println("No such file or directory: " + dataDirectory)
      System.exit(1)
    }
    if (d.exists && d.isFile) {
      System.out.println("Opening file " + dataDirectory)
      // Process the specified file.
      val rdd = sc.textFile(dataDirectory)
      val numlines = rdd.count()
      val maprd = rdd.map(str => MapRDBSpark.newDocument(str))
      val t0 = System.nanoTime * 1e-9
      maprd.saveToMapRDB(tableName, createTable = false, bulkInsert = false, idFieldPath = "id")
      val t = System.nanoTime() * 1e-9 - t0
      System.out.println("Persisted " + numlines + " records in " + t + " seconds")
    } else if (d.exists && d.isDirectory) {
      // Process any new files created in the specified directory.
      val recordsDStream = ssc.textFileStream(dataDirectory)
      recordsDStream.foreachRDD { rdd =>
        val numlines = rdd.count()
        val maprd = rdd.map(str => MapRDBSpark.newDocument(str))
        val t0 = System.nanoTime * 1e-9
        //TODO: validate RDD has field called "name"
        maprd.saveToMapRDB(tableName, createTable = false, bulkInsert = false, idFieldPath = "id")
        val t = System.nanoTime() * 1e-9 - t0
        if (numlines > 0)
          System.out.println("Persisted " + numlines + " records in " + t + " seconds")
      }
      System.out.println("Monitoring " + dataDirectory + " for new JSON files...")
      ssc.start()
      ssc.awaitTermination()
      ssc.stop(stopSparkContext = true, stopGracefully = true)
    }
  }
}
