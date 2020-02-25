# Benchmark simulating biological Neuron Model:
Multicore cpu and gpu Benchmark. But in fact it is simulation hodgkin huxley model neuron. Where for computing Ordinary Differential Equations i used Differentiable ODE solver method.
## First Version is as jupyter Notebook
Download or clone repository, open directory benchmark_notebook and open  bench.ipynb in jupyterLab in the Anaconda app.
## Script
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
## Plan
- Notebook cpu version (DONE)
- Support gpu (DONE)
- Shell Script (In Progress)
- App Linux
- App Windows
- ...
