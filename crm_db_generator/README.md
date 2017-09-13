# What does this do?

Generates a simulated CRM dataset using Logsynth. That dataset is saved as a JSON file. We then load it into Mapr-DB which we will use in another project for a customer 360 demo.

## Prerequisites

This utility uses the MapR-DB OJAI connector for Apache Spark, which requires the MapR Ecosystem Pack (MEP) 3.0 or later.
  
You'll also need to generate a JSON dataset. Log-synth is a good tool to use for that purpose. Here's how to install log-synth:

    $ git clone https://github.com/tdunning/log-synth
    $ mvn install -DskipTests

## How do I compile this project?

    $ mvn clean
    $ mvn package

## How do I run this project?

### Step 1 - Generate a customer directory dataset

    $ log-synth/target/log-synth -count 1M -schema crm_schema.json -format JSON > crm_data.json

### Step 2 - Start a filesystem watcher to load new JSON data into MapR-DB

    $ ssh nodea
    $ /opt/mapr/spark/spark-2.1.0/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar /mapr/demo.mapr.com/user/mapr/crm_data.json /tmp/crm_table

### Step 3 - Verify that the data is in MapR-DB

    $ mapr dbshell
      find /tmp/crm_data

### Step 4 - Generate a clickstream dataset

    $ cp data/urls.csv log-synth/src/resources/
    $ java -cp log-synth/target/log-synth-0.1-SNAPSHOT-jar-with-dependencies.jar com.mapr.synth.Synth -count 1M -schema data/clickstream_schema.json -format JSON > clickstream_data.json
