
from __future__ import print_function
"""
Created on Wed Apr 22 16:02:53 2015

Basic integrate-and-fire neuron with noise analysis
R Rao 2007

translated to Python by rkp 2015
Modified to analyze interspike interval distributions
"""
import numpy as np
import matplotlib.pyplot as plt

#Q11:
#input current
I = 10 # nA

#capacitance and leak resistance
C = 0.1 # nF
R = 100 # M ohms
tau = R*C # = 0.1*100 nF-Mohms = 100*100 pF Mohms = 10 ms
print('C = %.3f nF' % C)
print('R = %.3f M ohms' % R)
print('tau = %.3f ms' % tau)
print('(Theoretical)')

#membrane potential equation dV/dt = - V/RC + I/C

tstop = 150 # ms

V_inf = I*R # peak V (in mV)
tau = 0 # experimental (ms)

print('\n--- Steady State Calculation ---')
print('If current were never turned off:')
print('At steady state: dV/dt = 0')
print('From equation: dV/dt = -V/(RC) + I/C = 0')
print('Solving for V: V = I*R')
V_steady_state = I * R
print('V_steady = %.0f mV' % V_steady_state)
print('------------------------------------\n')

h = 0.2 #ms

V = 0 #mV
V_trace = [V] #in mV

for t in np.arange(h, tstop, h):

   #Euler method: V(t+h) = V(t) + h*dV/dt
   V = V +h*(- (V/(R*C)) + (I/C))

   if (not tau and (V > 0.6321*V_inf)):
     tau = t
     print('tau = %.3f ms' % tau)
     print('(Experimental)')


   if t >= 0.6*tstop:
     I = 0

   V_trace += [V]
   if t % 10 == 0:
       plt.plot(np.arange(0,t+h, h), V_trace, color='r')
       plt.xlim(0, tstop)
       plt.ylim(0, V_inf)
       plt.draw()
       
plt.show()







#Q16:
#capacitance and leak resistance
C = 1 #nF
R = 40
V_th = 10 

print("Integrate-and-Fire Neuron Threshold Analysis")
print("=" * 50)
print(f"Parameters:")
print(f"C = {C} nF")
print(f"R = {R} MΩ")
print(f"V_th = {V_th} mV")
print()

#At steady state: V_steady = I * R
#For threshold: I_threshold = V_th / R
I_threshold_theory = V_th / R  # nA
I_threshold_pA = I_threshold_theory * 1000  #pA

print("Theoretical Analysis:")
print(f"At steady state: V_steady = I × R")
print(f"For spiking: V_steady ≥ V_th")
print(f"Threshold current: I_th = V_th / R = {V_th} / {R} = {I_threshold_theory} nA")
print(f"I_th = {I_threshold_pA} pA")
print()
print(f"Largest current that FAILS to cause spiking: just below {I_threshold_pA} pA")
print()


def test_current(I_test):
    #Test if a given current causes spiking
    V = 0
    tstop = 500  
    abs_ref = 5
    ref = 0
    spike_count = 0
    
    for t in range(tstop):
        if not ref:
            V = V - (V/(R*C)) + (I_test/C)
        else:
            ref -= 1
            V = 0.2 * V_th
        
        if V > V_th:
            V = 50
            ref = abs_ref
            spike_count += 1
    
    return spike_count > 0, V if spike_count == 0 else "spiking"


print("Simulation Verification:")
test_currents = [0.20, 0.24, 0.249, 0.25, 0.251, 0.26, 0.30]  # nA

for I_test in test_currents:
    spikes, final_V = test_current(I_test)
    I_pA = I_test * 1000
    status = "SPIKES" if spikes else f"no spikes (V_final ≈ {final_V:.1f} mV)"
    print(f"I = {I_pA:4.0f} pA: {status}")

print()
print("ANSWER: 250 pA (rounded to nearest 10 pA)")


print("\n" + "="*50)
print("Original simulation with I = 1 nA:")

I = 1 #nA
V = 0
tstop = 200
abs_ref = 5
ref = 0
V_trace = []
V_th = 10

for t in range(tstop):
    if not ref:
        V = V - (V/(R*C)) + (I/C)
    else:
        ref -= 1
        V = 0.2 * V_th
    
    if V > V_th:
        V = 50
        ref = abs_ref

    V_trace += [V]

plt.figure(figsize=(10, 6))
plt.plot(V_trace)
plt.title(f'Integrate-and-Fire Neuron (I = {I} nA)')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.grid(True)
plt.show()





#Q18:
def simulate_neuron(noiseamp, tstop=5000):
   #para:
    I_base = 1  #
    C = 1  
    R = 40  
    V_th = 10  
    abs_ref = 5  
    
    V = 0
    ref = 0
    spiketimes = []
    
    I = I_base + noiseamp * np.random.normal(0, 1, tstop)
    
    for t in range(tstop):
        if not ref:
            V = V - (V/(R*C)) + (I[t]/C)
        else:
            ref -= 1
            V = 0.2 * V_th
        
        if V > V_th:
            spiketimes.append(t)
            V = 50
            ref = abs_ref
    
    return spiketimes

def analyze_interspike_intervals(spiketimes):
    if len(spiketimes) < 2:
        return []
    return np.diff(spiketimes)


noise_amplitudes = [0, 0.5, 1.0, 2.0, 3.0, 5.0]
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

plt.figure(figsize=(15, 10))

for i, noiseamp in enumerate(noise_amplitudes):
    print(f"Testing noise amplitude: {noiseamp} nA")
    
    spiketimes = simulate_neuron(noiseamp)
    intervals = analyze_interspike_intervals(spiketimes)
    
    if len(intervals) > 0:
        print(f"  Mean interval: {np.mean(intervals):.1f} ms")


