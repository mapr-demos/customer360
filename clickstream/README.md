# What does this do?

Generates a simulated clickstream dataset which is inserted into a MapR Streams topic and consumed by Spark. 

## How do I compile this project?

    $ mvn clean
    $ mvn package

## How do I run this project?

### Preliminary - create stream topics

Create a topic in MapR Streams for the clickstream:

    $ maprcli stream create -path /tmp/clickstream -produceperm p -consumeperm p -topicperm p
    $ maprcli stream topic create -path /tmp/clickstream -topic weblog
    
Run this command to delete that topic later:

    $ maprcli stream delete -path /tmp/clickstream

### Preliminary - synthesize a larger clickstream dataset (optional)

The clickstream dataset provided under data/ only has 10,000 rows. You can generate a larger clickstream with the following command. See also instructions in `crm_db_generator/README.md`.
    
    $ git clone https://github.com/tdunning/log-synth
    $ cp data/urls.csv log-synth/src/main/resources/
    $ cd log-synth
    $ mvn install -DskipTests
    $ cd ..
    $ log-synth/target/log-synth -count 1M -schema data/clickstream_schema.json -format JSON > clickstream_data.json
 
### Step 1 - Produce the clickstream dataset into a MapR Streams topic 

    $ java -cp target/mapr-sparkml-streaming-customer360-1.0.jar:`mapr classpath` com.mapr.demo.customer360.MsgProducer /tmp/clickstream:weblog ./data/clickstream_data.json

### Step 2 - Consume the clickstream dataset for analysis in Spark

    $ /opt/mapr/spark/spark-2.1.0/bin/spark-submit --class com.mapr.demo.customer360.ClickstreamConsumer --master local[2] target/mapr-sparkml-streaming-uber-1.0.jar /tmp/clickstream:weblog

