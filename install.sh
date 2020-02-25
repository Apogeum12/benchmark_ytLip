#!/bin/bash
echo "====> Install venv ."
sudo apt install python3-venv

echo "====> Create Environment .."
sudo python3 -m venv benchmark_script

echo "====> Activate Environment ..."
sudo source benchmark_script/bin/activate
