package com.mapr.demo.customer360;

/** ****************************************************************************
 * PURPOSE:
 *
 * Print the rate at which messages are published to a topic in MapR Streams.
 *
 * AUTHOR:
 * Ian Downard, idownard@mapr.com
 *
 * ****************************************************************************/
public class Monitor {
    public static double print_status(long records_processed, long startTime) {
        long elapsedTime = System.nanoTime() - startTime;
        double tput = records_processed / ((double) elapsedTime / 1000000000.0);
        System.out.printf("Throughput = %.2f msgs/sec published. Total published = %d\n", tput, records_processed);
        return tput;
    }
}
