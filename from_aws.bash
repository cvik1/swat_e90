#!/bin/bash

# get the name and ip
aws="ec2-3-14-153-116.us-east-2.compute.amazonaws.com"

sudo scp -i "../cvik_aws_key1.pem" ec2-user@$aws:e90/nohup.out ./results_1.out

# sudo scp -i "../cvik_aws_key1.pem" ec2-user@$aws:e90/models/model_r661.2686468345071.data-00000-of-00001 .
# sudo scp -i "../cvik_aws_key1.pem" ec2-user@$aws:e90/models/model_r661.2686468345071.index .
# sudo scp -i "../cvik_aws_key1.pem" ec2-user@$aws:e90/models/model_r661.2686468345071.meta .

sudo scp -i "../cvik_aws_key1.pem" ec2-user@$aws:e90/test_results/test_0_235.44489294065986.csv .
sudo scp -i "../cvik_aws_key1.pem" ec2-user@$aws:e90/test_results/test_2_463.631312117762.csv .
sudo scp -i "../cvik_aws_key1.pem" ec2-user@$aws:e90/test_results/test_13_661.0593893962748.csv .

# sudo scp -i "../cvik_aws_key1.pem" c2-user@$aws:e90/test_results/
