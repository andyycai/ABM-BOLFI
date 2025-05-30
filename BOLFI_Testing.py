#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 10:20:00 2025

@author: Andy.
"""

#%% Agent-Based Model

import mesa
import networkx as nx
from enum import Enum


class State(Enum):
    """Define the possible states for agents:"""
    
    NEVERSMOKER = 0
    SMOKER = 1
    QUITTER = 2


class NSQ_Model_UK(mesa.Model):
    """Define the model-level actions as NSQ_Model:"""
    
    def __init__(
            self,
            # Set seed (if necessary):
            # seed = ,
            # Parameters to be calibrated (5):
            delta_N_to_S,
            delta_S_to_Q,
            delta_Q_to_S,
            beta_S_to_Q_due_to_N,
            beta_Q_to_S_due_to_S,
            # Embedded network's structure: (not specified now, specify when implementing)
            network_type,
            # Calibrated parameters (2):
            beta_N_to_S_due_to_S = 0.40719,
            beta_S_to_Q_due_to_Q = 0.35214,
            # Initial population parameters (%, NEVERSMOKER, SMOKER, QUITTER):
            # initial_NEVERSMOKER_pct = 0.374, # 37.4% in 1974
            initial_SMOKER_pct = 0.456, # 45.6% in 1974
            initial_QUITTER_pct = 0.170, # 17.0% in 1974
            # Embedded network's details: (not specified now, specify when implementing)
            **network_parameters
            ):
        """Initialise the NSQ_Model class:"""
        
        # Initialise Mesa's 'Model' class:
        super().__init__() # super().__init__(seed) if seed != None
        
        # Pass the values of spontaneous parameters to the model class which is under construction:
        self.delta_N_to_S = delta_N_to_S
        self.delta_S_to_Q = delta_S_to_Q
        self.delta_Q_to_S = delta_Q_to_S
        
        # Pass the values of interaction parameters to the model class which is under construction:
        self.beta_N_to_S_due_to_S = beta_N_to_S_due_to_S
        self.beta_S_to_Q_due_to_N = beta_S_to_Q_due_to_N
        self.beta_S_to_Q_due_to_Q = beta_S_to_Q_due_to_Q
        self.beta_Q_to_S_due_to_S = beta_Q_to_S_due_to_S
        
        # Define estimated interaction parameters (from Christakis):
        self.beta_N_to_S_due_to_S_est = 1 - (1 - beta_N_to_S_due_to_S * (1 - (1 - delta_N_to_S)**4.5)) ** (1/4.5)
        self.beta_S_to_Q_due_to_Q_est = 1 - (1 - beta_S_to_Q_due_to_Q * (1 - (1 - delta_S_to_Q)**4.5)) ** (1/4.5)
        
        # Pass the values of initial population parameters to the model class which is under construction:
        self.initial_NEVERSMOKER_pct = 1 - (initial_SMOKER_pct + initial_QUITTER_pct)
        self.initial_SMOKER_pct = initial_SMOKER_pct
        self.initial_QUITTER_pct = initial_QUITTER_pct
        
        # Pass the embedded network's details to the model class which is under construction:
        self.graph = network_type(**network_parameters)
        # self.graph = network_type(*network_parameters.values()) "when wrapping network parameters as a dictionary"
        self.grid = mesa.space.NetworkGrid(self.graph)
        
        # print(f"Graph has {len(self.graph.nodes())} nodes", flush = True) "testing the # of nodes loaded"
        
        # Adding agents to nodes:
        for node in self.graph.nodes(): # [!] self.graph.nodes(): an iterable NodeView, not a list
            agent = NSQ_Agent(self, State.NEVERSMOKER) # Mesa 3 migration: no need to pass unique_id explicitly
            # self.schedule.add(agent) # Mesa 3 migration: eliminating the need to manually call ~"
            self.grid.place_agent(agent, node) # [!] Mesa automatically assigns agent.pos = node
            # when calling agent class, the objects in agent class are automatically passed
            # (agent, node) = (i+1, i), i starts from 0
            # [!] implicitly guarantee # of agents and # of nodes are equal
        
        # Randomly choose initial SMOKER from all agents (choose nodes first):
        initial_SMOKER_nodes = self.random.sample(list(self.graph.nodes()), round(self.initial_SMOKER_pct * len(list(self.graph.nodes()))))
        initial_SMOKER_agents = self.grid.get_cell_list_contents(initial_SMOKER_nodes)
        for agent in initial_SMOKER_agents:
            agent.current_state = State.SMOKER
            agent.updated_state = State.SMOKER
        
        # Randomly choose initial QUITTER from all remaining agents (choose nodes first):
        initial_QUITTER_nodes = self.random.sample(
            list(filter(lambda node: node not in initial_SMOKER_nodes, list(self.graph.nodes()))),
            round(self.initial_QUITTER_pct * len(list(self.graph.nodes())))
            )
        initial_QUITTER_agents = self.grid.get_cell_list_contents(initial_QUITTER_nodes)
        for agent in initial_QUITTER_agents:
            agent.current_state = State.QUITTER
            agent.updated_state = State.QUITTER
        
        # Generate initial NEVERSMOKER by deleting initial SMOKER and initial QUITTER:
        initial_NEVERSMOKER_nodes = list(filter(lambda node: node not in initial_SMOKER_nodes and node not in initial_QUITTER_nodes, list(self.graph.nodes())))
        initial_NEVERSMOKER_agents = self.grid.get_cell_list_contents(initial_NEVERSMOKER_nodes)
        
        # Add DataCollector to track desired data:
        self.datacollector = mesa.DataCollector(
            model_reporters={
                # Agents' list in each state: [!] collecting this causes the execution to be extremely slow
                # "NEVERSMOKER_agents": lambda m: m.get_agents_by_state(State.NEVERSMOKER),
                # "SMOKER_agents": lambda m: m.get_agents_by_state(State.SMOKER),
                # "QUITTER_agents": lambda m: m.get_agents_by_state(State.QUITTER),
                
                # Node lists of agents in each state:
                "NEVERSMOKER_nodes": lambda m: m.get_nodes_by_state(State.NEVERSMOKER),
                "SMOKER_nodes": lambda m: m.get_nodes_by_state(State.SMOKER),
                "QUITTER_nodes": lambda m: m.get_nodes_by_state(State.QUITTER),
        
                # Counts of agents in each state:
                "NEVERSMOKER_count": lambda m: m.count_agents_by_state(State.NEVERSMOKER),
                "SMOKER_count": lambda m: m.count_agents_by_state(State.SMOKER),
                "QUITTER_count": lambda m: m.count_agents_by_state(State.QUITTER),
                
                # Percentages of agents in each state:
                "% NEVERSMOKER": lambda m: m.pct_agents_by_state(State.NEVERSMOKER),
                "% SMOKER": lambda m: m.pct_agents_by_state(State.SMOKER),
                "% QUITTER": lambda m: m.pct_agents_by_state(State.QUITTER)
            },
            agent_reporters={
                "State": lambda a: a.current_state, # "current_state.name"
                "Node": lambda a: a.pos, # "pos"
            }
        )
        
        # Capture desired data from initial settings (step 0):
        self.datacollector.collect(self)
    
    def get_agents_by_state(self, state):
        """Return list of agents in a given state:"""
        return list(filter(lambda agent: agent.current_state is state, self.agents))
    
    def get_nodes_by_state(self, state):
        """Return list of nodes where their agents in a given state:"""
        return list(map(lambda agent: agent.pos, self.get_agents_by_state(state)))
    
    def count_agents_by_state(self, state):
        """Calculate counts of agents in a given state:"""
        return len(self.get_agents_by_state(state))

    def pct_agents_by_state(self, state):
        """Calculate percentage of agents in a given state:"""
        return (self.count_agents_by_state(state) / len(self.agents)) * 100
    
    def step(self):
        """Advance model by one step:"""
        self.agents.do("step")
        self.agents.do("advance")
        self.datacollector.collect(self)
        
    def run_model(self, n):
        """Run model for desired number of iterations"""
        for _ in range(n):
            self.step()
        
    def get_pct_data(self):
        """Returns percentage data by state:"""
        data = self.datacollector.get_model_vars_dataframe()
        pct_data = data[["% NEVERSMOKER", "% SMOKER", "% QUITTER"]]
        pct_data.index = range(1974, 1974 + len(pct_data)) # index starts from 1974
        years_to_omit = list(range(1975, 2000, 2)) # [1975, 1977, ..., 1999]
        pct_data = pct_data.drop(index = years_to_omit)
        return pct_data.round(1)
    
    def print_pct_data(self):
        """Print percentage data by state:"""
        print(self.get_pct_data())
        

class NSQ_Agent(mesa.Agent):
    """Define the agent-level actions as NSQ_Agent:"""
    
    def __init__(
            self,
            model,
            initial_state
            ):
        """Initialise the NSQ_Agent class:"""
        
        # Initialise Mesa's 'Agent' class and pass objects from the 'model' class to the agent class which is under construction:
        super().__init__(model)

        # Define current state before state change and initialise:
        self.current_state = initial_state
        
        # Define updated state after state change and initialise:
        self.updated_state = initial_state
        
    def NEVERSMOKER_initiation(self):
        """Define the dynamics of smoking initiation for NEVERSMOKER:"""
        
        # Identify neighbouring agents and those who are SMOKER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        SMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.SMOKER, neighbours))
        # SMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.SMOKER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_N_to_S:
            self.updated_state = State.SMOKER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            # prob_N_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_N_to_S_due_to_S)**len(SMOKER_neighbours))
            prob_N_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_N_to_S_due_to_S_est)**len(SMOKER_neighbours))
            if self.random.random() < prob_N_to_S_due_to_S:
                self.updated_state = State.SMOKER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, SMOKER_neighbours
    
    def SMOKER_cessation(self):
        """Define the dynamics of smoking cessation for SMOKER:"""
        
        # Identify neighbouring agents and those who are NEVERSMOKER and QUITTER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        NEVERSMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.NEVERSMOKER, neighbours))
        # NEVERSMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.NEVERSMOKER)]
        QUITTER_neighbours = list(filter(lambda agent: agent.current_state is State.QUITTER, neighbours))
        # QUITTER_neighbours = [agent for agent in neighbours if (agent.current_state is State.QUITTER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_S_to_Q:
            self.updated_state = State.QUITTER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            prob_S_to_Q_due_to_N = (len(NEVERSMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_N)**len(NEVERSMOKER_neighbours))
            # prob_S_to_Q_due_to_Q = (len(QUITTER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_Q)**len(QUITTER_neighbours))
            prob_S_to_Q_due_to_Q = (len(QUITTER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_S_to_Q_due_to_Q_est)**len(QUITTER_neighbours))
            if self.random.random() < prob_S_to_Q_due_to_N:
                self.updated_state = State.QUITTER
            elif self.random.random() < prob_S_to_Q_due_to_Q:
                self.updated_state = State.QUITTER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, NEVERSMOKER_neighbours, QUITTER_neighbours
    
    def QUITTER_relapse(self):
        """Define the dynamics of smoking relapse for QUITTER:"""
        
        # Identify neighbouring agents and those who are QUITTER:
        neighbours = self.model.grid.get_neighbors(self.pos, include_center = False)
        SMOKER_neighbours = list(filter(lambda agent: agent.current_state is State.SMOKER, neighbours))
        # SMOKER_neighbours = [agent for agent in neighbours if (agent.current_state is State.SMOKER)]
        
        # Define how state change is occured:
        # Spontaneity-based state change:
        if self.random.random() < self.model.delta_Q_to_S:
            self.updated_state = State.SMOKER
        # Interaction-based state change:
        elif len(neighbours) != 0:
            prob_Q_to_S_due_to_S = (len(SMOKER_neighbours) / len(neighbours)) * (1 - (1-self.model.beta_Q_to_S_due_to_S)**len(SMOKER_neighbours))
            if self.random.random() < prob_Q_to_S_due_to_S:
                self.updated_state = State.SMOKER
            else:
                self.updated_state = self.current_state
        # Otherwise:
        else:
            self.updated_state = self.current_state
        
        # Delete locally used objects to avoid confusion:
        # del neighbours, SMOKER_neighbours

    def step(self):
        """Decide what updated state should be for one iteration:"""
        
        # Smoking initiation for NEVERSMOKER:
        if self.current_state is State.NEVERSMOKER:
            self.NEVERSMOKER_initiation()
        
        # Smoking cessation for SMOKER:
        elif self.current_state is State.SMOKER:
            self.SMOKER_cessation()
        
        # Smoking relapse for QUITTER:
        elif self.current_state is State.QUITTER:
            self.QUITTER_relapse()
    
    def advance(self):
        """Apply any state change decided in step() to the agent's current state:"""
        self.current_state = self.updated_state

#%%

import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

# %matplotlib inline
# %precision 2

import logging
logging.basicConfig(level = logging.INFO)

import elfi

#%% Simulation Functions

def simulator_fc_UK(delta_N_to_S_fc_UK, delta_S_to_Q_fc_UK, delta_Q_to_S_fc_UK,
                    beta_SN_to_QN_fc_UK, beta_QS_to_SS_fc_UK,
                    timesteps = 49, # 1974-2023 (excluding the initial settings)
                    batch_size = 1, random_state = None
                    ):
    """"Define the NSQ_Model simulator on complete graph (fully-connected)"""
    model_fc_UK = NSQ_Model_UK(
        delta_N_to_S = delta_N_to_S_fc_UK,
        delta_S_to_Q = delta_S_to_Q_fc_UK,
        delta_Q_to_S = delta_Q_to_S_fc_UK,
        beta_S_to_Q_due_to_N = beta_SN_to_QN_fc_UK,
        beta_Q_to_S_due_to_S = beta_QS_to_SS_fc_UK,
        network_type = nx.complete_graph,
        # network_parameters
        n = 1000
        # network_parameters = {
        #     "n": 1000
        # }
    )
    
    model_fc_UK.run_model(timesteps)
    
    return model_fc_UK.get_pct_data()

def simulator_er_UK(delta_N_to_S_er_UK, delta_S_to_Q_er_UK, delta_Q_to_S_er_UK,
                    beta_SN_to_QN_er_UK, beta_QS_to_SS_er_UK,
                    timesteps = 49, # 1974-2023 (excluding the initial settings)
                    batch_size = 1, random_state = None
                    ):
    """"Define the NSQ_Model simulator on Erdös-Rényi graph"""
    model_er_UK = NSQ_Model_UK(
        delta_N_to_S = delta_N_to_S_er_UK,
        delta_S_to_Q = delta_S_to_Q_er_UK,
        delta_Q_to_S = delta_Q_to_S_er_UK,
        beta_S_to_Q_due_to_N = beta_SN_to_QN_er_UK,
        beta_Q_to_S_due_to_S = beta_QS_to_SS_er_UK,
        network_type = nx.erdos_renyi_graph,
        # network_parameters
        n = 1000,
        p = 0.003
        # network_parameters = {
        #     "n": 1000
        #     "p": 0.003
        # }
    )
    
    model_er_UK.run_model(timesteps)
    
    return model_er_UK.get_pct_data()

def simulator_ba_UK(delta_N_to_S_ba_UK, delta_S_to_Q_ba_UK, delta_Q_to_S_ba_UK,
                    beta_SN_to_QN_ba_UK, beta_QS_to_SS_ba_UK,
                    timesteps = 49, # 1974-2023 (excluding the initial settings)
                    batch_size = 1, random_state = None
                    ):
    """"Define the NSQ_Model simulator on Barabási-Albert network model"""
    model_ba_UK = NSQ_Model_UK(
        delta_N_to_S = delta_N_to_S_ba_UK,
        delta_S_to_Q = delta_S_to_Q_ba_UK,
        delta_Q_to_S = delta_Q_to_S_ba_UK,
        beta_S_to_Q_due_to_N = beta_SN_to_QN_ba_UK,
        beta_Q_to_S_due_to_S = beta_QS_to_SS_ba_UK,
        network_type = nx.barabasi_albert_graph,
        # network_parameters
        n = 1000,
        m = 2
        # network_parameters = {
        #     "n": 1000
        #     "m": 
        # }
    )
    
    model_ba_UK.run_model(timesteps)
    
    return model_ba_UK.get_pct_data()

def simulator_ws_UK(delta_N_to_S_ws_UK, delta_S_to_Q_ws_UK, delta_Q_to_S_ws_UK,
                    beta_SN_to_QN_ws_UK, beta_QS_to_SS_ws_UK,
                    timesteps = 49, # 1974-2023 (excluding the initial settings)
                    batch_size = 1, random_state = None
                    ):
    """"Define the NSQ_Model simulator on Watts-Strogatz network model"""
    model_ws_UK = NSQ_Model_UK(
        delta_N_to_S = delta_N_to_S_ws_UK,
        delta_S_to_Q = delta_S_to_Q_ws_UK,
        delta_Q_to_S = delta_Q_to_S_ws_UK,
        beta_S_to_Q_due_to_N = beta_SN_to_QN_ws_UK,
        beta_Q_to_S_due_to_S = beta_QS_to_SS_ws_UK,
        network_type = nx.watts_strogatz_graph,
        # network_parameters
        n = 1000,
        k = 4,
        p = 0.3
        # network_parameters = {
        #     "n": 1000
        #     "k": 
        #     "p": 
        # }
    )
    
    model_ws_UK.run_model(timesteps)
    
    return model_ws_UK.get_pct_data()

#%% Summary Statistics

def SMOKER_mean(df):
    return df["% SMOKER"].mean()

def SMOKER_median(df):
    return df["% SMOKER"].median()
    
def SMOKER_var(df):
    return df["% SMOKER"].var()

def SMOKER_dec_rate(df):
    return df["% SMOKER"].iloc[0] - df["% SMOKER"].iloc[-1]

def QUITTER_mean(df):
    return df["% QUITTER"].mean()

def QUITTER_median(df):
    return df["% QUITTER"].median()
    
def QUITTER_var(df):
    return df["% QUITTER"].var()

def QUITTER_incr_rate(df):
    return df["% QUITTER"].iloc[-1] - df["% QUITTER"].iloc[0]

# def summary_stats(df):
#     return np.array([
#         df["% SMOKER"].mean(), df["% SMOKER"].var(),
#         df["% QUITTER"].mean(), df["% QUITTER"].var()
#     ])

# Rolling mean and variance:

# def :
    
# def :


#%%

# 

observed_data = pd.read_excel("UK Data (1974-2023).xlsx", index_col = 0)

#%%

# Initialise the ELFI model:

# elfi_model_fc_UK = elfi.ElfiModel()

# Priors:

# delta_N_S_fc_UK = elfi.Prior("uniform", 0, 1e-4, model = elfi_model_fc_UK)
# delta_S_Q_fc_UK = elfi.Prior("uniform", 0, 1e-4, model = elfi_model_fc_UK)
# delta_Q_S_fc_UK = elfi.Prior("uniform", 0, 1e-4, model = elfi_model_fc_UK)
# beta_SN_QN_fc_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_fc_UK)
# beta_QS_SS_fc_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_fc_UK)

# Simulator node:

# sim_fc_UK = elfi.Simulator(simulator_fc_UK,
#                            delta_N_S_fc_UK, delta_S_Q_fc_UK, delta_Q_S_fc_UK,
#                            beta_SN_QN_fc_UK, beta_QS_SS_fc_UK,
#                            observed = observed_data,
#                            model = elfi_model_fc_UK)

# Summary statistics nodes:

# SMOKER_mean_fc_UK = elfi.Summary(SMOKER_mean, sim_fc_UK, model = elfi_model_fc_UK)
# SMOKER_var_fc_UK = elfi.Summary(SMOKER_var, sim_fc_UK, model = elfi_model_fc_UK)
# QUITTER_mean_fc_UK = elfi.Summary(QUITTER_mean, sim_fc_UK, model = elfi_model_fc_UK)
# QUITTER_var_fc_UK = elfi.Summary(QUITTER_var, sim_fc_UK, model = elfi_model_fc_UK)

# Distance node:

# d_fc_UK = elfi.Distance("euclidean",
#                         SMOKER_mean_fc_UK, SMOKER_var_fc_UK, QUITTER_mean_fc_UK, QUITTER_var_fc_UK,
#                         model = elfi_model_fc_UK)

# elfi.draw(elfi_model_fc_UK) # [!] need to debug

#%%

# Initialise the ELFI model:

elfi_model_er_UK = elfi.ElfiModel()

# Priors:
    
# delta_N_S_er_UK = 0
delta_N_S_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK) # 0
# delta_S_Q_er_UK = 0.031
delta_S_Q_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK) # 0.031
# delta_Q_S_er_UK = 0.0375
delta_Q_S_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK) # 0.0375
# beta_SN_QN_er_UK = 0.032
beta_SN_QN_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK) # 0.032
# beta_QS_SS_er_UK = 0.02
beta_QS_SS_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK) # 0.02

# delta_N_S_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK)
# delta_S_Q_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK)
# delta_Q_S_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK)
# beta_SN_QN_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK)
# beta_QS_SS_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model_er_UK)

# delta_N_S_er_UK = elfi.Prior("uniform", 0, 2e-4, model = elfi_model_er_UK) # 0 - 2e-4
# delta_S_Q_er_UK = elfi.Prior("uniform", 0, 0.1, model = elfi_model_er_UK) # 0 - 0.1
# delta_Q_S_er_UK = elfi.Prior("uniform", 0.01, 0.1, model = elfi_model_er_UK) # 0.01 - 0.1
# beta_SN_QN_er_UK = elfi.Prior("uniform", 0, 0.1, model = elfi_model_er_UK) # 0 - 0.1
# beta_QS_SS_er_UK = elfi.Prior("uniform", 0, 0.1, model = elfi_model_er_UK) # 0 - 0.1

# Simulator node:

sim_er_UK = elfi.Simulator(simulator_er_UK,
                           delta_N_S_er_UK, delta_S_Q_er_UK, delta_Q_S_er_UK,
                           beta_SN_QN_er_UK, beta_QS_SS_er_UK,
                           observed = observed_data,
                           model = elfi_model_er_UK)

# Summary statistics nodes:

SMOKER_mean_er_UK = elfi.Summary(SMOKER_mean, sim_er_UK, model = elfi_model_er_UK)
SMOKER_var_er_UK = elfi.Summary(SMOKER_var, sim_er_UK, model = elfi_model_er_UK)
SMOKER_dec_rate_er_UK = elfi.Summary(SMOKER_dec_rate, sim_er_UK, model = elfi_model_er_UK)
QUITTER_mean_er_UK = elfi.Summary(QUITTER_mean, sim_er_UK, model = elfi_model_er_UK)
QUITTER_var_er_UK = elfi.Summary(QUITTER_var, sim_er_UK, model = elfi_model_er_UK)
QUITTER_incr_rate_er_UK = elfi.Summary(QUITTER_incr_rate, sim_er_UK, model = elfi_model_er_UK)

# Distance node:

d_er_UK = elfi.Distance("euclidean",
                        SMOKER_mean_er_UK, SMOKER_var_er_UK,
                        QUITTER_mean_er_UK, QUITTER_var_er_UK,
                        SMOKER_dec_rate_er_UK, QUITTER_incr_rate_er_UK,
                        model = elfi_model_er_UK)

# 

log_d_er_UK = elfi.Operation(np.log, elfi_model_er_UK["d_er_UK"])

# 

bolfi_er_UK = elfi.BOLFI(log_d_er_UK,
                         batch_size = 1, # [!] In general, BOLFI does not benifit from a batch_size greater than 1
                         initial_evidence = 100, update_interval = 10,
                         # bounds = {
                                   # "delta_N_S_er_UK": (0, 1)}
                                   # "delta_S_Q_er_UK": (0, 1)}
                                   # "delta_Q_S_er_UK": (0, 1)}
                                   # "beta_SN_QN_er_UK": (0, 1)}
                                   # "beta_QS_SS_er_UK": (0, 1)}
                         bounds = {"delta_N_S_er_UK": (0, 1),
                                   "delta_S_Q_er_UK": (0, 1),
                                   "delta_Q_S_er_UK": (0, 1),
                                   "beta_SN_QN_er_UK": (0, 1),
                                   "beta_QS_SS_er_UK": (0, 1)}
                         # bounds = {"delta_N_S_er_UK": (0, 2e-4), # 0 - 2e-4
                         #           "delta_S_Q_er_UK": (0, 0.1), # 0 - 0.1
                         #           "delta_Q_S_er_UK": (0.01, 0.1), # 0.01 - 0.1
                         #           "beta_SN_QN_er_UK": (0, 0.1), # 0 - 0.1
                         #           "beta_QS_SS_er_UK": (0, 0.1)}, # 0 - 0.1
                         # bounds = {"delta_N_S_er_UK": (0, 1e-4), # 0 - 1e-4
                         #           "delta_S_Q_er_UK": (0.01, 0.05), # 0.01 - 0.05
                         #           "delta_Q_S_er_UK": (0.01, 0.05), # 0.01 - 0.05
                         #           "beta_SN_QN_er_UK": (0.01, 0.05), # 0.01 - 0.05
                         #           "beta_QS_SS_er_UK": (0.01, 0.05)}, # 0.01 - 0.05
                         # acq_noise_var = {
                         #                  "delta_N_S_er_UK": 0.1}
                         #                  "delta_S_Q_er_UK": 0.1}
                         #                  "delta_Q_S_er_UK": 0.1}
                         #                  "beta_SN_QN_er_UK": 0.1}
                         #                  "beta_QS_SS_er_UK": 0.1}
                         # acq_noise_var = {"delta_N_S_er_UK": 0.1,
                         #                  "delta_S_Q_er_UK": 0.1,
                         #                  "delta_Q_S_er_UK": 0.1,
                         #                  "beta_SN_QN_er_UK": 0.1,
                         #                  "beta_QS_SS_er_UK": 0.1}
                         )

posterior = bolfi_er_UK.fit(n_evidence = 200)

best_params = bolfi_er_UK.target_model.X[bolfi_er_UK.target_model.Y.argmin()]
best_discrepancy = bolfi_er_UK.target_model.Y.min()

print("Parameter order:", bolfi_er_UK.target_model.parameter_names)
print("Optimal parameters:", best_params)
print("Lowest discrepancy:", best_discrepancy)

result_bolfi_er_UK = bolfi_er_UK.sample(1000, info_freq=1000)
# result_bolfi_er_UK = bolfi_er_UK.sample(n_samples=10000, n_evidence=1000, algorithm='metropolis')
result_bolfi_er_UK.plot_traces()
result_bolfi_er_UK.plot_marginals()





