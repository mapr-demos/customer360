/* Copyright (c) 2009 & onwards. MapR Tech, Inc., All rights reserved */
package com.mapr.demo.customer360;
/** ****************************************************************************
 * PURPOSE:
 *
 * Produce clickstream data to a topic in MapR Streams. Limit streaming data to
 * user-specified msgs/sec throughput throttle (unlimited if not specified).
 *
 * USAGE:
 *
 * java -cp target/mapr-sparkml-streaming-customer360-1.0.jar:`mapr classpath` com.mapr.demo.customer360.MsgProducer stream:topic input_file [msgs/sec throttle]
 *
 * EXAMPLE:
 *
 * java -cp target/mapr-sparkml-streaming-customer360-1.0.jar:`mapr classpath` com.mapr.demo.customer360.MProducer /tmp/clickstream:weblog data/clickstream_data.json 5000
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
        Integer tput_throttle = Integer.MAX_VALUE;

        if (args.length == 3) {
            topic = args[0];
            fileName = args[1];
            tput_throttle = Integer.parseInt(args[2]);
        }
        System.out.println("Publishing to topic: "+ topic);
        configureProducer();
        System.out.println("Opening file " + args[1]);
        File f = new File(fileName);
        FileReader fr = new FileReader(f);
        BufferedReader reader = new BufferedReader(fr);
        String line = reader.readLine();
        long records_processed = 0L;

        try {
            long startTime = System.nanoTime();
            long last_update = 0;
            /* Add each message to a record. A ProducerRecord object
             identifies the topic or specific partition to publish
             a message to. */
            while (line != null) {
                ProducerRecord<String, String> rec = new ProducerRecord<>(topic, line);

                // Send the record to the producer client library.
                producer.send(rec);
                records_processed++;

                if (records_processed > tput_throttle) {
                    while ((Math.floor(System.nanoTime() - startTime) / 1e9) <= last_update) {
                        Thread.sleep(250); // Sleep for 250ms
                    }
                }

                // Print performance stats once per second
                if ((Math.floor(System.nanoTime() - startTime) / 1e9) > last_update) {
                    last_update++;
                    producer.flush();
                    Monitor.print_status(records_processed, startTime);
                }
                //System.out.println("Sent message: " + line);
                line = reader.readLine();
            }

        } catch (Throwable throwable) {
            System.err.printf("%s", throwable.getStackTrace());
        } finally {
            producer.close();
            System.out.println("Published " + records_processed + " messages to stream.");
            System.out.println("Finished.");
        }

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
        props.put("auto.create.topics.enable", true);

        producer = new KafkaProducer<>(props);
    }

}
