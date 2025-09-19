"""
This file has two separate problems:

Q6 — Firing rate vs. t_peak
- Run a simple neuron model driven by an alpha-function synaptic input.
- Sweep different values of the alpha function’s t_peak (the rise-to-peak time).
- Count output spikes and convert to firing rate.
- Plot how firing rate changes with t_peak.

Q9 — Steady-state network solution
- Define input weights (W), an input vector (u), and recurrent connections (M).
- Compute the steady-state output v_ss = (I - M)^(-1) · W · u.
- Verify the solution, check stability (eigenvalues of M), and compare with a
  few multiple-choice answer vectors to see which one matches.
"""


"""
Modified version to analyze t_peak vs firing rate relationship
Based on the alpha_neuron code from the course materials
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt



#Q6:
def run_neuron_simulation(t_peak_value):
    
    
    np.random.seed(0) 
    

    h = 1.
    t_max = 200 
    tstop = int(t_max/h) 
    ref = 0  
    
 
    thr = 0.9
    spike_train = np.random.rand(tstop) > thr
    
 
    t_a = 100  
    t_peak = t_peak_value #varying
    g_peak = 0.05  
    const = g_peak / (t_peak * np.exp(-1))
    t_vec = np.arange(0, t_a + h, h)
    alpha_func = const * t_vec * (np.exp(-t_vec/t_peak))
    
    #Neuron para:
    C = 0.5  
    R = 40
    
 
    g_ad = 0
    G_inc = 1/h
    tau_ad = 2
    
    #para:
    E_leak = -60 
    E_syn = 0
    g_syn = 0 
    V_th = -40 
    V_spike = 50 
    ref_max = 4/h  
    t_list = np.array([], dtype=int)
    V = E_leak
    

    output_spikes = 0
    

    for t in range(tstop):
        
    
        if spike_train[t]: 
            t_list = np.concatenate([t_list, [1]])
        
     
        g_syn = np.sum(alpha_func[t_list]) if len(t_list) > 0 else 0
        I_syn = g_syn * (E_syn - V)
        
     
        if np.any(t_list):
            t_list = t_list + 1
            if len(t_list) > 0 and t_list[0] == t_a: 
                t_list = t_list[1:]
        

        if not ref:
            V = V + h * (-((V - E_leak) * (1 + R * g_ad) / (R * C)) + (I_syn / C))
            g_ad = g_ad + h * (-g_ad / tau_ad)
        else:
            ref -= 1
            V = V_th - 10  
            g_ad = 0
        

        if (V > V_th) and not ref:
            V = V_spike
            ref = ref_max
            g_ad = g_ad + G_inc
            output_spikes += 1  
    
    return output_spikes


t_peak_values = np.arange(0.5, 10.5, 0.5)  # 0.5 to 10 ms in steps of 0.5 ms
firing_rates = []

print("Running simulations...")
for t_peak in t_peak_values:
    spike_count = run_neuron_simulation(t_peak)
    firing_rate = spike_count / (200/1000)  #spikes per second (200ms simulation)
    firing_rates.append(firing_rate)
    print(f"t_peak = {t_peak:.1f} ms: {spike_count} spikes, {firing_rate:.1f} Hz")

#results
plt.figure(figsize=(10, 6))
plt.plot(t_peak_values, firing_rates, 'bo-', linewidth=2, markersize=8)
plt.xlabel('t_peak (ms)', fontsize=12)
plt.ylabel('Firing Rate (Hz)', fontsize=12)
plt.title('Neuron Firing Rate vs. t_peak', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nFiring rate range: {min(firing_rates):.1f} - {max(firing_rates):.1f} Hz")
print("Relationship appears to be sublinear increase with t_peak")







#Q9:
import numpy as np

#matrices
W = np.array([[0.6, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.6, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.6, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.6, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.6]])

u = np.array([0.6, 0.5, 0.6, 0.2, 0.1])

M = np.array([[-0.25, 0, 0.25, 0.25, 0],
              [0, -0.25, 0, 0.25, 0.25],
              [0.25, 0, -0.25, 0, 0.25],
              [0.25, 0.25, 0.0, -0.25, 0],
              [0, 0.25, 0.25, 0, -0.25]])

print("W matrix:")
print(W)
print("\nInput vector u:")
print(u)
print("\nRecurrent weight matrix M:")
print(M)

I = np.eye(5) 
A = I - M 

print("\n(I - M) matrix:")
print(A)

#Check if (I - M) is invertible
det_A = np.linalg.det(A)
print(f"\nDeterminant of (I - M): {det_A:.6f}")

if abs(det_A) > 1e-10:
    A_inv = np.linalg.inv(A)
    print("\n(I - M)^(-1):")
    print(A_inv)
    
    #Calculate W*u first
    Wu = W @ u
    print("\nW*u:")
    print(Wu)
    
    #Calculate steady state: v_ss = (I - M)^(-1) * W * u
    v_ss = A_inv @ Wu
    print("\nSteady state output v_ss:")
    print(v_ss)
    
   
    verification = Wu + M @ v_ss
    print("\nVerification (should equal v_ss):")
    print(verification)
    print("\nDifference (should be near zero):")
    print(v_ss - verification)
    
else:
    print("Matrix (I - M) is not invertible - no unique steady state exists")


eigenvals, eigenvecs = np.linalg.eig(M)
print(f"\nEigenvalues of M:")
print(eigenvals)
print(f"Max absolute eigenvalue: {np.max(np.abs(eigenvals)):.6f}")


options = {
    "Option 1": np.array([1.67, 1.58, 1.66, 1.56, 1.53]),
    "Option 2": np.array([0.547, 0.480, 0.543, 0.381, 0.336]),
    "Option 3": np.array([0.873, 0.791, 0.864, 0.755, 0.718]),
    "Option 4": np.array([0.616, 0.540, 0.609, 0.471, 0.430])
}

print("\nComparison with given options:")
for name, option in options.items():
    diff = np.linalg.norm(v_ss - option)
    print(f"{name}: ||v_ss - option|| = {diff:.6f}")