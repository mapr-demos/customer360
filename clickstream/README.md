# What does this do?

Generates a simulated clickstream dataset which is inserted into a MapR Streams topic and consumed by Spark. 


## How do I compile this project?

mvn clean
mvn package

## How do I run this project?

### Step 1 - Produce the clickstream dataset into a MapR Streams topic 

    java -cp target/mapr-sparkml-streaming-customer360-1.0.jar:`mapr classpath` com.mapr.demo.customer360.MsgProducer /tmp/clickstream:401k data/cluster.txt

### Step 2 - Consume the clickstream dataset for analysis in Spark

    $ /opt/mapr/spark/spark-2.1.0/bin/spark-submit --class com.mapr.demo.customer360.ClickstreamConsumer --master local[2] target/mapr-sparkml-streaming-uber-1.0.jar /tmp/clickstream:401k

