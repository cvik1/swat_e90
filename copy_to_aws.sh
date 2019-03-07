#!/bin/bash

# get the name and ip 
aws = ""

scp -i ../cvik_aws_key1.pem ./AntAgents.py

scp -i ../cvik_aws_key1.pem ./train.py

scp -i ../cvik_aws_key1.pem instance_setup.sh
