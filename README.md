# Virus Optimization Algorithm (VOA)
This program is developed for implementing the Virus Optimization Algorithm (VOA).

## Function
1. The original version of continuous VOA.
2. The self-adaptive version of continuous VOA (SaVOA).

## Input Parameters
- func: The objective function (Default: None)
- dim: The number of dimension of solution space (Default: None)
- pop: The size of poppuation of virus (Default: 50 for original version)
- max_iter: The maximum number of iterations (Default: 1000)
- num_strong: The number of strong virus (Default: 10 for original version)
- gr_strong: The growth rate of strong virus (Default: 8 for original version) 
- gr_common: The growth rate of common virus (Default: 2 for original version) 
- bound: The boundary of solution space (Default: None)
- self-adaptive: Switch for original VOA and SaVOA (Default: False)
- show_train: Switch for showing the result of each iteration during training stage (Default: False) 

## How to use?
- Build a new instance from VOA class object
> voa_algo = VOA(func=func, dim=problem_size, pop=pop, max_iter=iterations, bound=bound, self_adaptive=False, show_train=False)
- Run VOA
> best_solution, best_fitness = voa_algo.run()

## Reference
- [1] Liang, Y. C., & Cuevas Juarez, J. R. (2016). A novel metaheuristic for continuous optimization problems: Virus optimization algorithm. Engineering Optimization, 48(1), 73-93. [[Access]](https://www.researchgate.net/profile/Yun-Chia_Liang/publication/276262183_A_novel_metaheuristic_for_continuous_optimization_problems_Virus_optimization_algorithm/links/5a43ba3f0f7e9ba868a77ee3/A-novel-metaheuristic-for-continuous-optimization-problems-Virus-optimization-algorithm.pdf)
- [2] Liang, Y. C., & Juarez, J. R. C. (2020). A self-adaptive virus optimization algorithm for continuous optimization problems. Soft Computing, 1-20. [[Access]](https://link.springer.com/article/10.1007%2Fs00500-020-04730-0)
