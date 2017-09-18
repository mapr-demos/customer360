/* Copyright (c) 2009 & onwards. MapR Tech, Inc., All rights reserved */
package com.mapr.demo.customer360;
/** ****************************************************************************
 * PURPOSE:
 *
 * Produce clickstream data to a topic in MapR Streams.
 *
 * USAGE:
 *
 * java -cp target/mapr-sparkml-streaming-customer360-1.0.jar:`mapr classpath` com.mapr.demo.customer360.MsgProducer /tmp/clickstream:401k data/cluster.txt
 *
 *
 * AUTHOR:
 * Ian Downard, idownard@mapr.com
 *
 * ****************************************************************************/
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;

import java.io.IOException;
import java.util.Properties;

public class MsgProducer {

    // Declare a new producer
    public static KafkaProducer producer;

    public static void main(String[] args) throws IOException {

        // Set the default data source and stream destination.
        String topic = "/tmp/clickstream:401k";
        String fileName = "data/clickstream_data.json";

        if (args.length == 2) {
            topic = args[0];
            fileName = args[1];

        }
        System.out.println("Sending to topic " + topic);
        configureProducer();
        File f = new File(fileName);
        FileReader fr = new FileReader(f);
        BufferedReader reader = new BufferedReader(fr);
        String line = reader.readLine();
        while (line != null) {
  
            /* Add each message to a record. A ProducerRecord object
             identifies the topic or specific partition to publish
             a message to. */
            ProducerRecord<String, String> rec = new ProducerRecord<>(topic,  line);

            // Send the record to the producer client library.
            producer.send(rec);
            System.out.println("Sent message: " + line);
            line = reader.readLine();

        }

        producer.close();
        System.out.println("All done.");

        System.exit(1);

    }

    /* Set the value for a configuration parameter.
     This configuration parameter specifies which class
     to use to serialize the value of each message.*/
    public static void configureProducer() {
        Properties props = new Properties();
        props.put("key.serializer",
                "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer",
                "org.apache.kafka.common.serialization.StringSerializer");

        producer = new KafkaProducer<>(props);
    }

}
