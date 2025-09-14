# AWS Batch Stock Feature Generator

This project applies **AWS Batch** to a real-world financial dataset to compute an "embarrassingly parallel" workload at scale.
The goal is to repeat the same core feature extraction task, but with varying arguments like start date and timeframe, thousands or millions of times in a timely manner.

In this setup, a custom worker function is executed, processing historical ticker data and logging any relevant metrics to a custom log group on AWS CloudWatch.
The resulting features can then be used for historical backtesting or model training. This codebase is meant to be used as a template for your own parallel workloads
and can be modified as needed.

## Assumptions

It is assumed you have an AWS account and are signed in through `aws sso login` on the command line to use `boto3`, after
configuring your credentials. Further, it is assumed the user has the necessary AWS Batch and S3 read/write permissions.

## Problem and Purpose

Running this workload sequentially—or even with local parallelism—would be prohibitively slow, potentially taking **days** to complete. By leveraging AWS Batch, we launch multiple EC2 instances in parallel, dramatically reducing runtime to **less than a day**.
The purpose of this project is to demonstrate how to use AWS Batch from the `boto3` SDK in Python. It gives a description of the important options and hopefully can act as a guide or even
a template for future use to run your own projects.

## Dataset

To follow the workflow of this project precisely, you must first create a bucket in which you would like to store the daily ticker data. I stored the OHLCV DataFrames
in a `dfs` folder, where each is saved as a Parquet file. Due to data subscriptions, I am unable to share the data, but you can easily re-create the task by pulling free
data from Yahoo Finance in Python.

- DataFrames stored in **Amazon S3** in a bucket with keys formatted as `dfs/<ticker_name>.parquet`
- My example contains **214 Parquet files**, each representing a ticker's DataFrame. Parquet is used for file compression and quicker reading functionality, since each file will need to be read multiple times.
- Each DataFrame has columns: `date`, `open`, `high`, `low`, `close`, `volume`
- Average coverage: ~12 years of daily data per ticker file, but trimmed down to ~9 years worth of data (2,213 unique trading days total) due to needing sufficient historical data to calculate some indicators.
- Starting date for processing: **2016-11-15**
- Run date: **2025-09-05**

## Task Breakdown

- For each ticker, process every day **on and after 2016-11-15**
- For each day, compute features across **5 timeframes** (`1` through `5`). `1` indicates daily bar data, `2` indicates 2-bar combined data, etc., up to 5-bar (weekly) bar data.
- Total days processed over all tickers: **473,582**
- Total tasks submitted: **473,582 × 5 = 2,367,910**

## Why AWS Batch?

- Handles massive parallelism with minimal orchestration  
- Scales EC2 compute dynamically  
- Avoids manual scheduling or local resource bottlenecks  
- Ideal for stateless, repeatable workloads like this one
- Provides a serverless option through Fargate, if desired.

## Usage

This repo includes:

- A Dockerized Python worker  
- Helper functions for S3 I/O  
- A job submission script using `boto3`  
- AWS Batch configuration templates
- A folder called `post_batch_analysis`, which analyzes worker performance and logs.

The folder `post_batch_analysis` relied on the logging structure and log group of the batch job to pull and report
statistics on those metrics using AWS queries in CloudWatch. After the job is done, run the script to determine
statistics on CPU and memory usage of your worker, which can help you to tune various Batch parameters for your
compute environment and job definitions.

Customize the worker logic and launch your own large-scale feature generation pipeline with minimal effort.

