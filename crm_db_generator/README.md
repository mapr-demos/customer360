# What does this do?

Generates a simulated CRM dataset using Logsynth. That dataset is saved as a JSON file. We then load it into Mapr-DB which we will use in another project for a customer 360 demo.

## Prerequisites

This utility uses the MapR-DB OJAI connector for Apache Spark, which requires the MapR Ecosystem Pack (MEP) 3.0 or later.
  
You'll also need to generate a JSON dataset. Log-synth is a good tool to use for that purpose. Here's how to install log-synth:

    $ git clone https://github.com/tdunning/log-synth
    $ cd log-synth 
    $ mvn install -DskipTests

## How do I compile this project?

    $ mvn clean
    $ mvn package

## How do I run this project?

### Step 1 - Generate a customer directory dataset

    $ log-synth/target/log-synth -count 1M -schema crm_schema.json -format JSON > crm_data.json
    $ scp crm_data.json nodea:~/

### Step 2 - Start a filesystem watcher to load new JSON data into MapR-DB

    $ ssh nodea
    $ /opt/mapr/spark/spark-2.1.0/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar ~/crm_data.json /tmp/crm_table

Another way to import JSON documents to a MapR-DB JSON table is to use the `mapr importJSON` command. So instead of the above spark command, you can use the following two commands to populate the CRM database table. Note, the importJSON command expects the source file to be in the MapR filesystem, so we move it there from the Linux filesystem using the command `hadoop fs -copyFromLocal`. This runs several seconds faster than the spark command, too.

    $ hadoop fs -copyFromLocal crm_data.json /user/mapr/
    $ /opt/mapr/bin/mapr importJSON -idField "id" -src /user/mapr/crm_data.json -dst /tmp/crm_table -mapreduce false

### Step 3 - Verify that the data is in MapR-DB

    $ mapr dbshell
      find /tmp/crm_data --limit 2

### Step 4 - Generate a clickstream dataset

    $ cp data/urls.csv log-synth/src/main/resources/
    $ cd log-synth
    $ mvn install -DskipTests
    $ cd ..
    $ log-synth/target/log-synth -count 1M -schema data/clickstream_schema.json -format JSON > clickstream_data.json 
