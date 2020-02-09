import torch
import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import argparse

from Neuron_Model import SNN

parser = argparse.ArgumentParser(description='Run benchmark')
parser.add_argument("--h", default=128, help='This is Hidden Size', type=int) # Hidden_Size 1-Layer Neuron --- My_propose: 1024 ---
#TODO: ADD +More argument + Add Block to function with seperate +  
args = parser.parse_args()

hidden_size = args.h
input_size=256    # Change value  --- My_propose: 1920x1080 = 2073600 ---
device = "cpu" # "cpu" -- Multicore + gpu  | "cuda" --- singlecore + gpu
x = torch.randn(input_size).to(device)
model = SNN(device, input_size, hidden_size).to(device)

test_loop = 10; test_score = list()
def test(): 
    for i in range(test_loop):
        start_time = time.time()
        X = model(x) #dopri5, euler, adams The best dopri5
        _time = time.time() - start_time
        test_score.append(_time)
    return test_score, X # Get Last X

test_score, X = test()
############## My cpu: intel i7 9750H ##############
print("--- Max: {:.4f} s --- Min: {:.4f} s --- Mean: {:.4f} s --- Error: {:.4f} s---".format(np.max(test_score)
                                                                                            ,np.min(test_score), sp.mean(test_score),
                                                                                            sp.std(test_score)))


time_step=60;dt=0.1
t = torch.arange(0.0, time_step, dt).to(device)
plt.subplots(figsize=(30, 7)) 
plt.title('Hodgkin-Huxley Neuron') 
plt.plot(t.cpu(), X[0].cpu(), 'c') #t, X[0]
plt.ylabel('V (mV)')
plt.xlabel('s (1e-1 s)')

plt.subplots(figsize=(30, 7)) 
plt.title('Hodgkin-Huxley Neuron') 
plt.plot(t.cpu(), X[1].cpu().detach().numpy(), 'k')
plt.plot(t.cpu(), X[2].cpu().detach().numpy(), 'c')
plt.plot(t.cpu(), X[3].cpu().detach().numpy(), 'b')
plt.ylabel('V (mV)')
plt.xlabel('s (1e-1 s)')
plt.show()