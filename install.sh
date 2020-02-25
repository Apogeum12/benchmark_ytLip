#!/bin/bash
echo "====> Install venv ."
sudo apt install python3-venv

echo "====> Create Environment .."
sudo python3 -m venv benchmark_script

echo "====> Activate Environment ..."
sudo source benchmark_script/bin/activate

echo "====> Install package ...."
sudo pip3 install torch torchvision \
            git+https://github.com/rtqichen/torchdiffeq \
            numpy  cycler==0.10.0 kiwisolver==1.1.0 \
            matplotlib==3.1.3 Pillow==7.0.0 pkg-resources==0.0.0 \
            pyparsing==2.4.6 python-dateutil==2.8.1 six==1.14.0
            
sudo apt install python3-tk  
 
echo "====> Done ....."

