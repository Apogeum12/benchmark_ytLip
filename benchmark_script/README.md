# Benchmark_script
Benchmark in script version
## Installation
For Linux Version:
- In Directly benchmark_ytLip/
```bash
sudo apt install python3-venv
python3 -m venv benchmark_script
source benchmark_script/bin/activate 
```
- Install backend for matplotlib
```bash
sudo apt install python3-tk
```
- Install Environment
```bash
pip3 install -r requirements.txt
```
### Run:
In directly folder benchmark_script
- [--h] Num Hiden Neuron; [--i] Num input Size; [--d] Device choice; [--t] Test Size
- If Dont put Neuron Hidden num then Hiden is equal 2 * sqrt(Input_size)
```bash
python3 bench.py --i 2048 --d cpu --t 10
```
