# Overview

This project contains a web app that represents a customer service portal containing various data visualizations for a customer 360 scenario. The web app and data visualizations are provided by the bokeh python library.

We are using Ian's Intel NUC devices to run this bokeh app.

# Preconditions

- git and maven must be installed. `sudo apt-get install git maven -y`
- virtualenv must be installed. `sudo apt-get install python-virtualenv -y`.
- clone the customer360 project to your cluster, with `git clone https://github.com/mapr-demos/customer360.git`.
- anaconda and bokeh must have been installed. For example, like this:

```
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash Anaconda3-4.4.0-Linux-x86_64.sh
. ~/.bashrc
conda install bokeh
```


## Install Mapr-ODBC driver

Install an ODBC driver and define a data source connectors for Drill. Run these commands, and refer to the sample `.ini` config files are included in `resources/`:

```
wget http://package.mapr.com/tools/MapR-ODBC/MapR_Drill/MapRDrill_odbc_v1.3.0.1009/maprdrill-1.3.0.1009-1.x86_64.rpm
rpm2cpio maprdrill-1.3.0.1009-1.x86_64.rpm | cpio -idmv
sudo mv opt/mapr/drill /opt/mapr/drillodbc
cd /opt/mapr/drillodbc/Setup
cp mapr.drillodbc.ini ~/.mapr.drillodbc.ini
cp odbc.ini ~/.odbc.ini
cp odbcinst.ini ~/.odbcinst.ini	

# Put these export commands in .bashrc, too!
export ODBCINI=~/.odbc.ini
export MAPRDRILLINI=~/.mapr.drillodbc.ini
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/opt/mapr/drillodbc/lib/64:/usr/lib64
```

For more informaiont on setting ODBC connectors, see [http://www.bigendiandata.com/2017-05-01-Apache_Drill](http://www.bigendiandata.com/2017-05-01-Apache_Drill/#connecting-to-drill-from-ubuntu), or the [Drill docs](https://drill.apache.org/docs/configuring-odbc-on-linux/#step-2:-define-the-odbc-data-sources-in-.odbc.ini).


# Create a master customer table in MapR-DB

Build the crm data generator tool:
```
cd customer360/crm_db_generator
mvn clean
mvn package
cd target
```

Then run these commands:

```
cp ../../bokeh/datasets/crm_data.json /mapr/my.cluster.com/user/mapr/
/opt/mapr/spark/spark-2.1.0/bin/spark-submit --master local[2] --class com.mapr.demo.customer360.LoadJsonToMaprDB mapr-demo-customer360-1.0-jar-with-dependencies.jar /mapr/my.cluster.com/user/mapr/crm_data.json /tmp/crm_data
```

Verify that the table was created by running this command:

```
mapr dbshell "find /tmp/crm_data --limit 2"
```

    
## Configure secondary indexes (optional)

To make it faster to load and filter the customer directory table, create secondary indexes for the email and phone number fields, like this:

```
$ maprcli table index add -path /tmp/crm_data -index idx_name -indexedfields '"name":-1' -includedfields '"name", "phone_number"'
$ maprcli table index add -path /tmp/crm_data -index idx_phone -indexedfields '"phone_number":-1' -includedfields '"name", "phone_number"'

```

You can verify that completed successfully by ensuring "email" and "name" are in the `coveredFields` attribute shown
by the following command:

```
$ sqlline

sqlline> !connect jdbc:drill:zk=localhost:5181
sqlline> explain plan for select name, phone_number, email from dfs.`/tmp/crm_data` where email = 'RoslynSolomon@example.com';
```
    
## Configure global query monitoring for Drill (optional)

Here is how to configure the Drill web console so it lists the queries executed by every drillbit service in a cluster, instead of just those on the local node: 

Add this line inside the "drill.exec" block in `/opt/mapr/drill/drill-1.11.0/conf/drill-override.conf` on every node where Drill is installed:

	sys.store.provider.zk.blobroot: "maprfs:///tmp/drill"

Then restart the drillbit service:

	/opt/mapr/drill/drill-1.11.0/bin/drillbit.sh restart

It could take up to 2 minutes before the Drill web console starts.  The Drill web console runs on port 8042 (e.g. [http://nodea:8042](http://nodea:8042).


# Start Bokeh

    ssh nodea
    # install prerequites for scipy (without these, you'll get an error on `pip install scipy`
    sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
    # install dependencies for pyodbc
    sudo apt-get install python-pip unixodbc-dev python-numpy rpm2cpio
    
    # create an isolated environment
    virtualenv env
    # enter the isolated environemnt
    source ~/env/bin/activate 
    
    # install python packages (this may not work, if it doesn't just install each one individually)
    cd customer360/bokeh/
    sudo pip install -r requirements.txt
    
    # make sure you have internet access so we can download datasets
    ping www.google.com
    # run bokeh
    cd ~/customer360/bokeh/
    bokeh serve . --log-level=debug --allow-websocket-origin '*'
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
     