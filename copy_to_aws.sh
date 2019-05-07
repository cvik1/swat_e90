#!/bin/bash

# get the name and ip
aws="ec2-18-191-204-52.us-east-2.compute.amazonaws.com"

sudo scp -i "../cvik_aws_key1.pem" ./scripts/AntAgents.py ec2-user@$aws:e90/.

sudo scp -i "../cvik_aws_key1.pem" ./scripts/train.py ec2-user@$aws:e90/.

sudo scp -i "../cvik_aws_key1.pem" ./scripts/ptrain.py ec2-user@$aws:e90/.

sudo scp -i "../cvik_aws_key1.pem" instance_setup.sh ec2-user@$aws:e90/.
