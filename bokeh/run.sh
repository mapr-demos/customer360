#!/bin/bash
cd ~/customer360/bokeh
export ODBCINI=/home/mapr/.odbc.ini
export MAPRDRILLINI=/home/mapr/.mapr.drillodbc.ini
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/opt/mapr/drillodbc/lib/64:/usr/lib64:/opt/mapr/lib:/usr/lib/jvm/java-7-openjdk-amd64/jre/lib/amd64/server/
bokeh serve . --allow-websocket-origin '*'
