BY USING THIS SOFTWARE, YOU EXPRESSLY ACCEPT AND AGREE TO THE TERMS OF THE AGREEMENT CONTAINED IN THIS GITHUB REPOSITORY.  See the file EULA.md for details.

#  Customer 360 Powered by MapR&trade;

The <a href="https://mapr.com/products/mapr-converged-data-platform/">MapR Converged Data Platform</a> is uniquely suited to run 
Customer 360 applications because it provides the foundations for

1. Cloud Scale Storage
2. Schemaless Data Integration
3. Machine Learning

Customer 360 applications require the ability to access data lakes containing structured and unstructured data, integrate data sets, and run operational and analytical workloads simultaneously. MapR enables applications to glean customer intelligence through machine learning that relates to customer personality, sentiment, propensity to buy, and likelihood to churn. Check out the <a href="https://mapr.com/solutions/quickstart/customer-360-knowing-your-customer-is-step-one/">Customer 360 Quick Start Solution</a> to learn more about MapR's products and solutions for Customer 360 applications.

This demo application focuses on showing how the following three tenants to customer 360 applications can be achieved on MapR:

1. Big Data storage of structured and semi-structured data in files, tables, and streams
2. SQL-based data integration of disperate datasets
3. Predictive analytics through machine learning insights

<img src="https://github.com/mapr-demos/customer360/blob/master/images/dataflow.gif" width="70%">

## Overview

There are a lot of different ways to demo MapR. Some people like to use data visualizations to convey the value of MapR's technology. Other people like go deep into APIs for more developer oriented conversations. In this demo, we try to accomodate both approaches. For the graphical approach, this demo application runs in a stand-alone web server that shows interactive data visualizations (shown below). For the deep technical dive, we provide Jupyter Notebooks to show code and MapR's APIs.

<img src="https://github.com/mapr-demos/customer360/blob/master/images/screenshot.png" width="70%">


# Demo Script

## How does "convergence" make Customer Intelligence better?

Company's have all kinds of information about their customers. That information is stored in many different formats and in many different ways. It may be schemaless or schemaful. It may be in database tables, it may be in files, or it may be in streams like web clicks or social media activity. In all these cases, MapR provides the platform you need to ingest and store the data. MapR enables you to conveniently ingest data by mounting the cluster with NFS. Furthermore, storing massive and unbounded data streams are no problem for MapR's **cloud scale data storage**.

But we don't want to just store data, we want to be able to access all this information in order to analyze customer behavior and product usage. So **Data Integration** is also a critical part for Customer 360. MapR does data integration with Apache Drill - the industry's best SQL engine for Hadoop. 

But its not enough to simply consolidate datasets. We want to do more than just SQL. We want to harness the power of **Machine Learning** to not only better understand customer characteristics like sentiment, propensity to buy, and likelihood to churn but also to improve fraud detection and targeted marketing.

Cloud Scale Storage, Data Integration, and Machine Learning are essential to achieving the most value out of a Customer 360 application.  Any one of those would be a challenge by itself, but with the MapR Converged Data Platform you get them all in a package that is faster/better/cheaper than anything else.

## Setting the Scene

You're a customer support representative. You're about to answer a call from Erika Gallardo. Before you talk to her try to answer these questions: 

- Is she already navigating the web site? 
- What should you keep in mind to upsell? 
- What do you think she'll be asking about?

## What's behind the scenes?

- **Customer Directory** is populated by a SQL query which accesses data in MapR-DB, MySQL, and JSON files. Thanks to Apache Drill, these SQL queries can be executed without specifying a schema.
- **Machine Learning** is used to predict spend rate and lifetime value by linear regressions. It's also used to create the Heatmap that identifies upsell/cross-sell opportunities for customer groups segemented by K-Means.
- **Real-Time Streaming** is used to show clickstreams for each user. This data helps customer service reps know how customers are using our bank's web site during support calls. Clickstream data is also useful for tracking the long-term customer journey.
- **Survey Feedback** is derived from surveys in which only a fraction of customers responded, but which we can extrapolate to identify patterns with other customers in the same group.
- **Customer Image** is an example of binary data saved as jpg files which can be used for fraud detection by camera surveillance in banks.


# Get Community Support!

Visit the [MapR Community](https://community.mapr.com/) pages where you can post questions and discuss your use case.





