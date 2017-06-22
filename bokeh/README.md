# Overview

This project contains a web app that represents a customer service portal containing various data visualizations for a customer 360 scenario. The web app and data visualizations are provided by the bokeh python library.

We are using Ian's Intel NUC devices to run this bokeh app.

# Preconditions

- The drill64 DSN must be defined on the bokeh host
- Data must have already been copied to /mapr/my.cluster.com/dialogue_corpus
- Data must have already been copied to /mapr/my.cluster.com/face_images

# Create the CRM table in MapR-DB

Copy the provided datasets/crm_data.json file to the cluster and create the CRM database like this:

    `/opt/mapr/spark/spark-2.0.1/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar /mapr/my.cluster.com/user/mapr/crm_data.json /tmp/crm_data`


# Start Bokeh

    ssh nodea
    # enter the isolated python virtualenv
    source ~/tmp/my_project/bin/activate 
    # make sure you have internet access so we can download datasets
    ping www.google.com
    # run bokeh
    ~/customer360/bin/bokeh serve . --log-level=debug --allow-websocket-origin '*'
    open http://nodea:5006/bokeh

# Start Jupyter

    # enter the isolated python virtualenv
    source ~/tmp/my_project/bin/activate 
    cd ~/customer360
    jupyter &
    http://nodea:8888

# Data Generation

## CRM dataset
    
Install logsynth, a tool for synthesizing data:

    git clone https://github.com/tdunning/log-synth
    mvn install -DskipTests

Generate a large JSON data containing CRM data

    ~/development/log-synth/target/log-synth -count 1M -schema crm_schema.json -format JSON > crm_data.json
    
    
Load CRM data into MapR-DB. With this utility, we'll just copy CRM data as a json file to a directory which is being watched for new files.  
    
    ssh nodea /opt/mapr/spark/spark-2.1.0/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar /mapr/demo.mapr.com/tmp /tmp/crm_data &
    scp crm_data.json nodea:/mapr/

Verify that the data is in MapR-DB (optional)
    
        mapr dbshell
        find /tmp/crm_data

# Installation Procedure for "MapR Sandbox for Hadoop 5.2.1"  

## (This doesn't work yet)

    yum update -y
    yum -y install yum-utils
    yum -y groupinstall development
    yum -y install https://centos7.iuscommunity.org/ius-release.rpm
    yum groupinstall -y 'development tools'
    yum install -y zlib-dev openssl-devel sqlite-devel bzip2-devel
    yum install xz-libs
    wget http://www.python.org/ftp/python/2.7.10/Python-2.7.10.tar.xz
    xz -d Python-2.7.10.tar.xz
    tar -xvf Python-2.7.10.tar
    cd Python-2.7.10
    ./configure --prefix=/usr/local
    make
    make altinstall
    ./python -V
    rpm -ivh http://dl.fedoraproject.org/pub/epel/6/i386/epel-release-6-8.noarch.rpm
    yum install -y python27
    yum install python
    yum install python-pip
    pip install --upgrade pip
    pip install virtualenv
   
    sudo yum update -y
    do this: https://gist.github.com/hangtwenty/5546945
    sudo yum install -y python-devel python-setuptools python-pip
    sudo pip install virtualenv
    cd ~
    virtualenv env1
    source ~/env1/bin/activate
    
# References:
https://gist.github.com/xuelangZF/570caf66cd1f204f98905e35336c9fc0
https://github.com/h2oai/h2o-2/wiki/installing-python-2.7-on-centos-6.3.-follow-this-sequence-exactly-for-centos-machine-only
https://www.godaddy.com/help/how-to-install-python-pip-on-centos-12367
     