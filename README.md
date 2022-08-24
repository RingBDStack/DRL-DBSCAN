# DRL-DBSCAN
Code for 'Automating DBSCAN via Reinforcement Learning' accepted by CIKM2022

# Overview
- main.py: getting started.
- model/model.py: overall framework of DRL-DBSCAN.
- model/environment.py: environment of the reinforcement learning process.
- model/TD3.py: TD3 structure.
- utils/utils.py: utils for data preprocessing, loading and other functions.
- utils/plot.py: visual functions

# Datasets
the experimental dataset
consists of 4 artificial clustering benchmark datasets and 1 public real-world streaming dataset. 
The benchmark datasets are the 2D shape sets, including: Aggregation, Compound, Pathbased, and D31. 
They involve multiple density types such as clusters within clusters, multi-density, multi-shape,
closed loops, etc., and have various data scales. Furthermore, the real-world streaming dataset Sensor comes from consecutive
information (temperature, humidity, light, and sensor voltage) collected from 54 sensors deployed by the Intel Berkeley Research Lab.
We use a subset of 80, 864 for experiments and divide these objects
into 16 data blocks (V1, ..., V16) as an online dataset.

# Baselines
We compare proposed DRL-DBSCAN
with three types of baselines: (1) traditional hyperparameter search
CIKM’22, October 17-22, 2022, Hybrid Conference, Hosted in Atlanta, Georgia, USA Ruitong Zhang, et al.
schemes: random search algorithm Rand, Bayesian optimization based on Tree-structured Parzen estimator algorithm BO-TPE;
 (2) meta-heuristic optimization algorithms: the simulated annealing optimization Anneal, particle swarm optimization
PSO, genetic algorithm GA, and differential evolution
algorithm DE; (3) existing DBSCAN parameter search methods: KDist (V-DBSCAN) and BDE-DBSCAN

# Citation
If you find this repository helpful, please consider citing the following paper.
