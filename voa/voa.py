#！/usr/bin/python
# -*- coding: utf-8 -*-#


'''
---------------------------------
 Name:         voa
 Description:  
 Author:       Samuel Li, Ray Li
 Date:         21/06/2020
---------------------------------
'''


import numpy as np
import random
import math
from .virus import Virus


class VOA:

    def __init__(self, func=None, dim=None, pop=50, max_iter=1000, num_strong=10, gr_strong=8, gr_common=2, bound=None,
                 self_adaptive=False, show_train=False):
        self.func = func
        self.dim = dim
        self.bound = bound
        self.show_train = show_train
        self.iterations = max_iter
        self.fitness_lst = []
        self.self_adaptive = self_adaptive
        print("Self_adaptive: {}".format(self.self_adaptive))

        self.solu_collection = []
        self.viruses = []
        # VOA para
        if self.self_adaptive:
            self.init_pop = 2
            self.cur_pop = self.init_pop
            self.strong_pop = 1
            self.growth_rate_strong = 1
            self.growth_rate_common = 1
            self.max_pop = 1000
            self.t = 1 / ((2 * self.dim) ** 0.5)
            self.t_ = 1 / ((2 * (self.dim ** 0.5)) ** 0.5)

        else:
            self.init_pop = pop  # Number of initial solutions (Viruses)
            self.strong_pop = num_strong    # Number of strong viruses
            self.growth_rate_strong = gr_strong    # The growth rate for generating new solution from the strong viruses.
            self.growth_rate_common = gr_common    # The growth rate for generating new solution from the common viruses.
            self.intensity = 1
            self.max_pop = 1000

        self.replication_counter = 0
        self.avg_fitness = None
        self.best_solu = {'fitness': float('inf'), 'structure': None, 'type:': None}

    def classification(self):
        fitness_lst = []
        for _, virus_ind in enumerate(self.viruses):
            if virus_ind.fitness is None:
                virus_ind.fitness = self.func(virus_ind.structure)
            fitness_lst.append(virus_ind.fitness)

        fitness_lst = np.array(fitness_lst)
        sorted_idx = np.argsort(fitness_lst)    # The index of virus fitness sorted by ascending order
        strong_idx = sorted_idx[:self.strong_pop]

        for idx, virus_ind in enumerate(self.viruses):
            if idx in strong_idx:
                virus_ind.type = 'strong'
            else:
                virus_ind.type = 'common'

    def replication(self):

        new_member = []
        for _, virus_ind in enumerate(self.viruses):
            structure = virus_ind.structure
            if virus_ind.type == 'strong':
                # Strong virus
                for idx in range(self.growth_rate_strong):
                    new_structure = []
                    if self.self_adaptive:
                        delta_x = virus_ind.sigma * np.random.randn(self.dim) * structure
                        new_structure = structure + delta_x
                    else:

                        for dim_i in range(self.dim):
                            rand = (np.random.rand() * 2) - 1   # [0, 1) => [-1, 1)
                            nv = structure[dim_i] + ((rand/self.intensity) * structure[dim_i])
                            new_structure.append(nv)
                        new_structure = np.array(new_structure)
                    new_structure = np.clip(new_structure, *self.bound)
                    new_virus = Virus(dim=self.dim, bound=self.bound, structure=new_structure, type='new')
                    # Evaluate objective function value
                    solution = new_virus.structure
                    new_virus.fitness = self.func(solution)
                    new_member.append(new_virus)

            elif virus_ind.type == 'common':
                # Common virus
                for idx in range(self.growth_rate_common):
                    new_structure = []
                    if self.self_adaptive:
                        delta_x = virus_ind.sigma * np.random.randn(self.dim) * structure
                        new_structure = structure + delta_x
                    else:
                        for dim_i in range(self.dim):
                            rand = (np.random.rand() * 2) - 1   # [0, 1) => [-1, 1)
                            nv = structure[dim_i] + (rand * structure[dim_i])
                            new_structure.append(nv)
                        new_structure = np.array(new_structure)
                    new_structure = np.clip(new_structure, *self.bound)
                    new_virus = Virus(dim=self.dim, bound=self.bound, structure=new_structure, type='new')
                    # Evaluate objective function value
                    solution = new_virus.structure
                    new_virus.fitness = self.func(solution)
                    new_member.append(new_virus)
            else:
                print("Incorrect virus type, please check the code. [{}]".format(virus_ind.type))
                raise

        # Combine
        self.viruses = self.viruses + new_member

    def evaluate_performance(self):
        fitness_lst = []
        for _, virus_ind in enumerate(self.viruses):
            fitness_lst.append(virus_ind.fitness)
        avg_fitness = np.mean(np.array(fitness_lst))
        return avg_fitness

    def antivirus_mechanism(self, cur_avg_fitness):
        pop_size = len(self.viruses)
        amount = np.random.randint(low=1, high=(pop_size - self.strong_pop))    # Amount of viruses killed
        worse_lst = []
        better_lst = []
        fitness_lst = []
        for idx, virus_ind in enumerate(self.viruses):
            if virus_ind.fitness > cur_avg_fitness:
                worse_lst.append(idx)
            else:
                better_lst.append(idx)
            fitness_lst.append(virus_ind.fitness)
        if amount > len(worse_lst):
            # Randomly add virus from better list to worse list
            add_amount = amount - len(worse_lst)
            worse_lst = worse_lst + random.sample(better_lst, add_amount)
        else:
            # Keep the first {amount} worse viruses selected by fitness value on worse list.
            fitness_lst = np.array(fitness_lst)
            sorted_idx = np.argsort(fitness_lst)
            sorted_idx_desc = sorted_idx[::-1]
            worse_lst = sorted_idx_desc[:amount]

        survival_viruses = []
        fitness_lst = []
        for idx, virus_ind in enumerate(self.viruses):
            if idx not in worse_lst:
                survival_viruses.append(virus_ind)
                fitness_lst.append(virus_ind.fitness)
        self.viruses = survival_viruses
        return fitness_lst, amount

    def stop_criteria(self):
        pass

    def run(self):

        self.viruses = []
        # Initialize viruses
        for _ in range(self.init_pop):
            # The structure of virus will determine the location of virus inside the host cell.
            structure = np.array(
                [self.bound[0] + np.random.rand() * (self.bound[1] - self.bound[0]) for k in range(self.dim)])
            virus_ind = Virus(dim=self.dim, bound=self.bound, structure=structure)
            # Evaluate objective function value
            solution = virus_ind.structure
            virus_ind.fitness = self.func(solution)
            self.viruses.append(virus_ind)

        # Iteration
        for iteration in range(self.iterations):
            self.replication_counter = self.replication_counter + 1
            self.classification()   # Classify the type of viruses
            self.replication()  # Generate new virus members
            cur_avg_fitness = self.evaluate_performance()   # Evaluate average objective function value
            if not self.self_adaptive:
                if self.avg_fitness is None:
                    self.avg_fitness = cur_avg_fitness
                else:
                    if cur_avg_fitness >= self.avg_fitness:
                        # No improvement, and update the intensity factor.
                        self.intensity = self.intensity + 1
                    else:
                        # Performance improved
                        self.avg_fitness = cur_avg_fitness

            # Antivirus mechanism
            before_pop = len(self.viruses)
            cur_fitness_lst, amount_killed = self.antivirus_mechanism(cur_avg_fitness)
            cur_fitness_lst = np.array(cur_fitness_lst)
            sorted_idx = np.argsort(cur_fitness_lst)
            best_virus = self.viruses[sorted_idx[0]]
            # limit the population of virus
            if len(self.viruses) > self.max_pop:
                new_viruses = []
                survival_idx = sorted_idx[:self.init_pop]   # Keep the virus who has lower fitness value
                for idx, virus_ind in enumerate(self.viruses):
                    if idx in survival_idx:
                        new_viruses.append(virus_ind)
                self.viruses = new_viruses

            # Update self-adaptive parameters
            if self.self_adaptive:
                # Update the perturbation
                for _, virus_ind in enumerate(self.viruses):
                    virus_ind.update_sigma(t=self.t, t_=self.t_)

                # Update the number of strong virus
                delta_s = 1 + ((len(self.viruses) - self.cur_pop)/(max(len(self.viruses), self.cur_pop)))
                self.strong_pop = math.ceil(self.strong_pop * delta_s)
                self.cur_pop = len(self.viruses)
                # print("Num of cur_pop: {}, Num of strong_pop: {}".format(self.cur_pop, self.strong_pop))

                # Update the growth rate
                cur_fitness_lst = []
                for _, virus_ind in enumerate(self.viruses):
                    cur_fitness_lst.append(virus_ind.fitness)
                cur_fitness_lst = np.array(cur_fitness_lst)
                sorted_idx = np.argsort(cur_fitness_lst)
                strong_idx = sorted_idx[:self.strong_pop]
                common_idx = sorted_idx[self.strong_pop:]
                # Strong growth rate
                strong_flag = True  # flag 0
                for idx in strong_idx:
                    if self.viruses[idx].type != 'strong':
                        strong_flag = False     # flag 1
                if strong_flag:
                    # delta_strong = 1 + (amount_killed/len(self.viruses))
                    delta_strong = 1 + (amount_killed / before_pop)
                    self.growth_rate_strong = math.ceil(self.growth_rate_strong * delta_strong)
                    if self.growth_rate_strong >= 10:
                        # print("-------num: {} , amount_kill: {}, pop: {}, strong".format(self.growth_rate_strong,
                        #                                                             amount_killed, len(self.viruses)))
                        self.growth_rate_strong = 1
                # Common growth rate
                common_flag = True
                for idx in common_idx:
                    if self.viruses[idx].type != 'common':
                        common_flag = False
                if common_flag:
                    # delta_common = 1 + (amount_killed/len(self.viruses))
                    delta_common = 1 + (amount_killed / before_pop)
                    self.growth_rate_common = math.ceil(self.growth_rate_common * delta_common)
                    if self.growth_rate_common >= 10:
                        # print("-------num: {} , amount_kill: {}, pop: {}, common".format(self.growth_rate_common,
                        #                                                            amount_killed, len(self.viruses)))
                        self.growth_rate_common = 1
                # print("Growth rate: strong->{}, common=>{}".format(self.growth_rate_strong, self.growth_rate_common))

            # Save the best solution
            if best_virus.fitness < self.best_solu['fitness']:
                self.best_solu['fitness'] = best_virus.fitness
                self.best_solu['structure'] = best_virus.structure
                self.best_solu['type'] = best_virus.type

            if self.show_train:
                print('Iteration: {}, fitness value: {}'.format(iteration, self.best_solu['fitness']))

            self.fitness_lst.append(self.best_solu['fitness'])
            if (iteration % 10) == 0:
                sig_iter_solu = []
                pop_idx = sorted_idx[:self.init_pop]
                for idx, virus_ind in enumerate(self.viruses):
                    if idx in pop_idx:
                        solu = list(virus_ind.structure[:2])
                        solu.append(virus_ind.type)
                        sig_iter_solu.append(solu)
                self.solu_collection.append(sig_iter_solu)

            # Evaluate stopping criterion
            # self.stop_criteria()

        return self.best_solu['structure'], self.best_solu['fitness']

    def all_hisotry(self):
        return self.fitness_lst

