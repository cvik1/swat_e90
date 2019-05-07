#!/bin/bash

sudo yum update

source activate tensorflow_p36

mkdir e90/models

mkdir e90/test_results

pip install gym

pip install pybullet

cd /home/ec2-user

git clone https://github.com/benelot/pybullet-gym.git

cd pybullet-gym

pip install -e .

cd /home/ec2-user

conda config --add channels conda-forge

conda config --add channels npbool

conda install ffmpeg

conda install libav

cd /home/ec2-user



#xvfb-run -s "-screen 0 640x480x24" nohup python train.py -r -n 100000 &
