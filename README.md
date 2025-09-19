Computational Neuroscience Coursework

This repository contains my exercises and quiz work for computational neuroscience courses. It includes implementations of foundational models of memory, learning, and decision-making.

Contents

* Intro to Comp Neuro quiz work My quiz solutions and related work (certificate excluded).
* Perceptual-Decision-Making Exercises inspired by Wong & Wang (2006). These simulate competing neural populations that integrate noisy evidence and make a choice when one population wins. - `Wong-Wang decision making network.py`  
*   Core network setup (parameters, dynamics, helpers) for the Wong–Wang model.
* 
* - `individual decision trials.py`  
*   Runs **single-trial** simulations for selected coherence levels; plots synaptic traces, firing rates, competition (r1–r2), and phase portraits; reports choice and decision time.
* 
* - `psychophysical curves.py`  
*   Runs **many trials per coherence** and plots:
*   psychometric (accuracy vs coherence), chronometric (RT vs coherence), speed–accuracy scatter, RT distributions, and a simple confidence proxy (|s1–s2|).
* 
* - `fundamental trade-off between speed … .py`  
*   Explores the **speed–accuracy trade-off** by manipulating task/decision parameters (e.g., decision threshold/urgency/baseline) and visualizes how faster responding reduces accuracy.
* 
* - `decision outcomes.py`  
*   Summarizes **choice outcomes** across conditions (e.g., winner population, accuracy by coherence/threshold) and can produce histograms or tables of result counts.
*   
* Hopfield Network.py Recurrent neural network model of associative memory. Stores multiple patterns as attractors and recalls them from partial/noisy input.
* Oja's learning rule.py A stabilized version of Hebbian learning. Prevents weight explosion and extracts the first principal component of the input, showing a connection between synaptic plasticity and dimensionality reduction.
* Spatial Working Memory.py A bump attractor network (Compte et al., 2000) modeling how populations of neurons in prefrontal cortex can sustain persistent activity and maintain information in working memory.

Summary

Together, these exercises cover:
* Memory → Hopfield Network & Spatial Working Memory
* Learning → Oja’s Learning Rule
* Decision-Making → Wong-Wang model of perceptual choice
This collection highlights fundamental computational principles used to understand how the brain learns, remembers, and makes decisions.
