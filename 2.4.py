#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 4 10:20:00 2026

@author: Andy.
"""

#%%

import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import pandas as pd

import mesa
import networkx as nx
from enum import Enum

# %matplotlib inline
# %precision 2

import logging
logging.basicConfig(level = logging.INFO)

import elfi

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

#%% Data

observed_data = pd.read_excel("UK Data (1974-2023).xlsx", index_col = 0)

#%%

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

#%% Summary Statistics & Distance Measure

# Differences by Cells:

def differences_by_cells(simulated_df, observed_df):
    """Compute cell differences (+/-) between DataFrames and return a 1-d NumPy array"""
    differences_df = simulated_df - observed_df # differences matrix (DataFrame)
    return differences_df.values.flatten() # convert DataFrame to NumPy 1-d array (12,)

def df_to_array(df):
    """Convert a simulated DataFrame to a 2-d NumPy array"""
    return df.values.reshape(1, -1) # convert DataFrame to NumPy 2-d array (1,12)

def euclidean_dist_by_cells(simulated_df, observed_df):
    """Compute the Euclidean distance between a simulated data and the observed data"""
    differences_df = simulated_df - observed_df # differences matrix (DataFrame)
    differences_array = differences_df.values.reshape(1, -1) # convert DataFrame to NumPy 2-d array (1,12)
    return np.linalg.norm(differences_array, ord = 2)

# Relative Differences:

def relative_differences(df):
    """Compute the differences by cells in each column (times series)"""
    relative_differences_df = df.diff().dropna() # compute differences and remove rows with NaN
    relative_differences_df.index = [f"{i}/{j}" for i, j in zip(df.index[:-1], df.index[1:])] # change index; e.g. 2022/2023
    # return relative_differences_df
    return relative_differences_df.values.reshape(1, -1) # convert DataFrame to NumPy 2-d array (1,9)

# Moments Method: (Michael has confirmed that no explicit advantages would be brought by using moments method)

def moments(df):
    """a"""


#%% ELFI & BOLFI (Erdös-Rényi Graph)

# Initialise the ELFI model:

elfi_model = elfi.ElfiModel()

# Prior nodes:
    
delta_N_S_er_UK = 0.00075
delta_S_Q_er_UK = 0.031
delta_Q_S_er_UK = 0.04
# beta_SN_QN_er_UK = 0.032
beta_SN_QN_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model)
# beta_QS_SS_er_UK = 0.02
beta_QS_SS_er_UK = elfi.Prior("uniform", 0, 1, model = elfi_model)

# Simulator node:

simulator = elfi.Simulator(simulator_er_UK,
                           delta_N_S_er_UK,
                           delta_S_Q_er_UK,
                           delta_Q_S_er_UK,
                           beta_SN_QN_er_UK,
                           beta_QS_SS_er_UK,
                           observed = observed_data,
                           model = elfi_model)

# Summary Statistics nodes:

summary_stats = elfi.Summary(relative_differences,
                             # df_to_array,
                             simulator,
                             model = elfi_model)

# Distance node:

disc = elfi.Distance("euclidean", summary_stats,
                     model = elfi_model)

# Taking a logarithm of the discrepancies to reduce the effect that high discrepancies have on the Gaussian Process (GP) surrogate model:
# NOTE: log_d can be very negative or even -inf

log_disc = elfi.Operation(np.log, elfi_model["disc"])

# BOLFI & its Gaussian Process (GP) surrogate model:

bolfi = elfi.BOLFI(elfi_model, # model
                   log_disc, # target_name
                   batch_size = 1, # [!] In general, BOLFI does not benifit from a batch_size greater than 1
                   initial_evidence = 100, update_interval = 1,
                   bounds = {"beta_SN_QN_er_UK": (0, 1),
                             "beta_QS_SS_er_UK": (0, 1)
                             },
                   acq_noise_var = {"beta_SN_QN_er_UK": 0.1,
                                    "beta_QS_SS_er_UK": 0.1
                                    }
                   )

bolfi_posterior_fit = bolfi.fit(n_evidence = 500) # posterior: fit()
bolfi_posterior_ep = bolfi.extract_posterior() # posterior: extract_posterior()
bolfi.plot_state()
bolfi.plot_discrepancy()

# Optimal parameters and log-discrepancy (target_model / extract_result()): target_model

bolfi_results_tm = bolfi.target_model

# print(dir(bolfi_results_tm))
# target_model.X: evaluated parameter points; target_model.Y: observed discrepancies

optimal_parameters_tm = bolfi_results_tm.X[bolfi_results_tm.Y.argmin()]
optimal_log_disc_tm = bolfi_results_tm.Y.min()

print("Parameters:", bolfi_results_tm.parameter_names)
print("Optimal values:", optimal_parameters_tm)
print("Optimal log-discrepancy:", optimal_log_disc_tm)

# Optimal parameters and log-discrepancy (target_model / extract_result()): extract_result()

bolfi_results_er = bolfi.extract_result()

# optimal_parameters_er_dir = bolfi_results_er.x_min # [!] very wrong?
# print("Optimal parameters:", optimal_parameters_er_dir) # [!] very wrong?

# print(dir(bolfi_results_er))
# print(bolfi_results_er.outputs)

optimal_log_disc_er_indir = bolfi_results_er.outputs["log_disc"].min()
optimal_infos_no_er_indir = bolfi_results_er.outputs["log_disc"].argmin()
optimal_parameters_er_indir = list(bolfi_results_er.outputs[parameter][optimal_infos_no_er_indir]
                                   for parameter in bolfi_results_er.parameter_names)
print("Parameters:", bolfi_results_er.parameter_names)
print("Optimal values:", optimal_parameters_er_indir)
print("Optimal log-discrepancy:", optimal_log_disc_er_indir)

# Log-discrepancy against Parameter Value plots:

log_disc_v_beta_x, log_disc_v_beta_y = np.meshgrid(np.linspace(*bolfi_results_tm.bounds[0],
                                                               101), # default num = 50
                                                   np.linspace(*bolfi_results_tm.bounds[1],
                                                               101)) # default num = 50

bolfi_log_disc_predicted_mean = (np.vectorize(lambda a, b: bolfi_results_tm.predict(np.array([a, b]))))(log_disc_v_beta_x, log_disc_v_beta_y)[0]
bolfi_log_disc_predicted_var = (np.vectorize(lambda a, b: bolfi_results_tm.predict(np.array([a, b]))))(log_disc_v_beta_x, log_disc_v_beta_y)[1]
bolfi_log_disc_predicted_lower_bound = bolfi_log_disc_predicted_mean - 1.645 * np.sqrt(bolfi_log_disc_predicted_var) # 0.1 quantile
bolfi_log_disc_predicted_upper_bound = bolfi_log_disc_predicted_mean + 1.645 * np.sqrt(bolfi_log_disc_predicted_var) # 0.9 quantile

bolfi_log_disc_predicted_all_values = np.concatenate([bolfi_log_disc_predicted_mean, bolfi_log_disc_predicted_lower_bound, bolfi_log_disc_predicted_upper_bound])

# bolfi_log_disc_v_plots, (bolfi_log_disc_predicted_mean_contour_plot,
#                          bolfi_log_disc_predicted_surfaces_plot) = plt.subplots(1, 2, figsize = (15, 6))

bolfi_log_disc_v_plots = plt.figure(figsize = (15, 6))

# Log-discrepancy against Parameter Value plot (customised contour plot with contourf()):

bolfi_log_disc_predicted_mean_contour_plot = bolfi_log_disc_v_plots.add_subplot(1, 2, 1)
bolfi_log_disc_predicted_mean_contour = bolfi_log_disc_predicted_mean_contour_plot.contourf(log_disc_v_beta_x, log_disc_v_beta_y, bolfi_log_disc_predicted_mean, levels = 30, cmap = "cividis", alpha = 0.7)
bolfi_log_disc_predicted_mean_contour_plot.set_title("Predicted log-discrepancy mean")
bolfi_log_disc_predicted_mean_contour_plot.set_xlabel(bolfi_results_tm.parameter_names[0])
bolfi_log_disc_predicted_mean_contour_plot.set_ylabel(bolfi_results_tm.parameter_names[1])
bolfi_log_disc_predicted_mean_contour_plot_cbar = bolfi_log_disc_v_plots.colorbar(bolfi_log_disc_predicted_mean_contour, ax = bolfi_log_disc_predicted_mean_contour_plot, label = "log-discrepancy")

# Log-discrepancy against Parameter Value plot (3-d surfaces plot with plot_surface() and contour()):

bolfi_log_disc_predicted_surfaces_plot = bolfi_log_disc_v_plots.add_subplot(1, 2, 2, projection = "3d")
bolfi_log_disc_predicted_surfaces_plot.set_proj_type("ortho")
# bolfi_log_disc_predicted_surfaces_plot.grid(False)

bolfi_log_disc_predicted_surfaces_plot_base_cmap = cm.cividis
# bolfi_log_disc_predicted_surfaces_plot_cmap_mean = bolfi_log_disc_predicted_surfaces_plot_base_cmap(np.linspace(0.33, 0.66, 100))
# bolfi_log_disc_predicted_surfaces_plot_cmap_lower_bound = bolfi_log_disc_predicted_surfaces_plot_base_cmap(np.linspace(0.0, 0.33, 100))
# bolfi_log_disc_predicted_surfaces_plot_cmap_upper_bound = bolfi_log_disc_predicted_surfaces_plot_base_cmap(np.linspace(0.66, 1.0, 100))

# bolfi_log_disc_predicted_norm_mean = plt.Normalize(vmin = bolfi_log_disc_predicted_mean.min(), vmax = bolfi_log_disc_predicted_mean.max())
# bolfi_log_disc_predicted_norm_lower_bound = plt.Normalize(vmin = bolfi_log_disc_predicted_lower_bound.min(), vmax = bolfi_log_disc_predicted_lower_bound.max())
# bolfi_log_disc_predicted_norm_upper_bound = plt.Normalize(vmin = bolfi_log_disc_predicted_upper_bound.min(), vmax = bolfi_log_disc_predicted_upper_bound.max())
bolfi_log_disc_predicted_norm = plt.Normalize(vmin = bolfi_log_disc_predicted_all_values.min(), vmax = bolfi_log_disc_predicted_all_values.max())

bolfi_log_disc_predicted_mean_surface = bolfi_log_disc_predicted_surfaces_plot.plot_surface(log_disc_v_beta_x, log_disc_v_beta_y,
                                                                                            bolfi_log_disc_predicted_mean,
                                                                                            # facecolors = bolfi_log_disc_predicted_surfaces_plot_cmap_mean,
                                                                                            # facecolors = bolfi_log_disc_predicted_surfaces_plot_base_cmap(0.33 + 0.33 * bolfi_log_disc_predicted_norm_mean(bolfi_log_disc_predicted_mean)),
                                                                                            facecolors = bolfi_log_disc_predicted_surfaces_plot_base_cmap(bolfi_log_disc_predicted_norm(bolfi_log_disc_predicted_mean)),
                                                                                            alpha = 0.7,
                                                                                            edgecolor = "none",
                                                                                            label = "mean")
bolfi_log_disc_predicted_lower_bound_surface = bolfi_log_disc_predicted_surfaces_plot.plot_surface(log_disc_v_beta_x, log_disc_v_beta_y,
                                                                                                   bolfi_log_disc_predicted_lower_bound,
                                                                                                   # facecolors = bolfi_log_disc_predicted_surfaces_plot_cmap_lower_bound,
                                                                                                   # facecolors = bolfi_log_disc_predicted_surfaces_plot_base_cmap(0.33 * bolfi_log_disc_predicted_norm_lower_bound(bolfi_log_disc_predicted_lower_bound)),
                                                                                                   facecolors = bolfi_log_disc_predicted_surfaces_plot_base_cmap(bolfi_log_disc_predicted_norm(bolfi_log_disc_predicted_lower_bound)),
                                                                                                   alpha = 0.7,
                                                                                                   edgecolor = "none",
                                                                                                   label = "0.1 quantile")
bolfi_log_disc_predicted_upper_bound_surface = bolfi_log_disc_predicted_surfaces_plot.plot_surface(log_disc_v_beta_x, log_disc_v_beta_y,
                                                                                                   bolfi_log_disc_predicted_upper_bound,
                                                                                                   # facecolors = bolfi_log_disc_predicted_surfaces_plot_cmap_upper_bound,
                                                                                                   # facecolors = bolfi_log_disc_predicted_surfaces_plot_base_cmap(0.66 + 0.34 * bolfi_log_disc_predicted_norm_upper_bound(bolfi_log_disc_predicted_upper_bound)),
                                                                                                   facecolors = bolfi_log_disc_predicted_surfaces_plot_base_cmap(bolfi_log_disc_predicted_norm(bolfi_log_disc_predicted_upper_bound)),
                                                                                                   alpha = 0.7,
                                                                                                   edgecolor = "none",
                                                                                                   label = "0.9 quantile")
bolfi_log_disc_predicted_mean_surface_contour = bolfi_log_disc_predicted_surfaces_plot.contour(log_disc_v_beta_x, log_disc_v_beta_y, bolfi_log_disc_predicted_mean,
                                                                                               zdir = "z", offset = None, levels = 30, cmap = bolfi_log_disc_predicted_surfaces_plot_base_cmap, alpha = 1)
bolfi_log_disc_predicted_lower_bound_surface_contour = bolfi_log_disc_predicted_surfaces_plot.contour(log_disc_v_beta_x, log_disc_v_beta_y, bolfi_log_disc_predicted_lower_bound,
                                                                                                      zdir = 'z', offset = None, levels = 30, cmap = bolfi_log_disc_predicted_surfaces_plot_base_cmap, alpha = 1)
bolfi_log_disc_predicted_upper_bound_surface_contour = bolfi_log_disc_predicted_surfaces_plot.contour(log_disc_v_beta_x, log_disc_v_beta_y, bolfi_log_disc_predicted_upper_bound,
                                                                                                      zdir = 'z', offset = None, levels = 30, cmap = bolfi_log_disc_predicted_surfaces_plot_base_cmap, alpha = 1)

bolfi_log_disc_predicted_surfaces_plot.set_title("Predicted log-discrepancy")
bolfi_log_disc_predicted_surfaces_plot.set_xlabel(bolfi_results_tm.parameter_names[0])
bolfi_log_disc_predicted_surfaces_plot.set_ylabel(bolfi_results_tm.parameter_names[1])
# bolfi_log_disc_predicted_surfaces_plot.set_zlabel("log-discrepancy")
bolfi_log_disc_predicted_surfaces_plot.legend()

# bolfi_log_disc_predicted_surfaces_plot_cbar_sm = plt.cm.ScalarMappable(cmap = bolfi_log_disc_predicted_surfaces_plot_base_cmap)
# bolfi_log_disc_predicted_surfaces_plot_cbar_sm.set_array(bolfi_log_disc_predicted_all_values)
bolfi_log_disc_predicted_surfaces_plot_cbar_sm = plt.cm.ScalarMappable(cmap = bolfi_log_disc_predicted_surfaces_plot_base_cmap, norm = bolfi_log_disc_predicted_norm)
bolfi_log_disc_predicted_surfaces_plot_cbar = bolfi_log_disc_v_plots.colorbar(bolfi_log_disc_predicted_surfaces_plot_cbar_sm, ax = bolfi_log_disc_predicted_surfaces_plot, label = "log-discrepancy")

plt.tight_layout()
plt.show()

# Posterior (fit() / extract_posterior()):

# bolfi_posterior_fit = bolfi.fit(n_evidence = 200)
# bolfi_posterior_ep = bolfi.extract_posterior()

# print(dir(bolfi_posterior_fit))
# print(dir(bolfi_posterior_fit.model))
# print(dir(bolfi_posterior_fit.model.bounds))
# print(bolfi_posterior_fit.model.bounds[0]) # bounds for parameter_names[0]
# print(bolfi_posterior_fit.model.bounds[1]) # bounds for parameter_names[1]

# print(dir(bolfi_posterior_ep))
# print(dir(bolfi_posterior_ep.model))
# print(dir(bolfi_posterior_ep.model.bounds))
# print(bolfi_posterior_ep.model.bounds[0]) # bounds for parameter_names[0]
# print(bolfi_posterior_ep.model.bounds[1]) # bounds for parameter_names[1]

# Posterior's pdf & logpdf plots:

# Posterior's pdf & logpdf plots (contour plots with fit() / plot()):

# bolfi_posterior_fit.plot(logpdf = False) # (x_axis, y_axis) = (parameter_names)
# bolfi_posterior_fit.plot(logpdf = True) # (x_axis, y_axis) = (parameter_names)

# Posterior's pdf & logpdf plots (contour plots with extract_posterior() / plot()):

# bolfi_posterior_ep.plot(logpdf = False) # (x_axis, y_axis) = (parameter_names)
# bolfi_posterior_ep.plot(logpdf = True) # (x_axis, y_axis) = (parameter_names)

# Posterior's pdf & logpdf plots (customised contour plots with contour() & contourf()):

# posterior_beta_x, posterior_beta_y = np.meshgrid(np.linspace(0, 1, 101), # default num = 50
#                                                  np.linspace(0, 1, 101)) # default num = 50

posterior_beta_x, posterior_beta_y = np.meshgrid(np.linspace(*bolfi_posterior_ep.model.bounds[0],
                                                             101), # default num = 50
                                                 np.linspace(*bolfi_posterior_ep.model.bounds[1],
                                                             101)) # default num = 50

# bolfi_posterior_pdf_default = (np.vectorize(lambda a, b: bolfi_posterior_fit.pdf(np.array([a, b]))))(posterior_beta_x, posterior_beta_y)
# bolfi_posterior_pdf_customised = bolfi_posterior_fit.pdf(np.vstack([posterior_beta_x.ravel(), posterior_beta_y.ravel()]).T).reshape(len(posterior_beta_x), len(posterior_beta_y))
# bolfi_posterior_logpdf_default = (np.vectorize(lambda a, b: bolfi_posterior_fit.logpdf(np.array([a, b]))))(posterior_beta_x, posterior_beta_y)
# bolfi_posterior_logpdf_customised = bolfi_posterior_fit.logpdf(np.vstack([posterior_beta_x.ravel(), posterior_beta_y.ravel()]).T).reshape(len(posterior_beta_x), len(posterior_beta_y))

bolfi_posterior_pdf_default = (np.vectorize(lambda a, b: bolfi_posterior_ep.pdf(np.array([a, b]))))(posterior_beta_x, posterior_beta_y)
bolfi_posterior_pdf_customised = bolfi_posterior_ep.pdf(np.vstack([posterior_beta_x.ravel(), posterior_beta_y.ravel()]).T).reshape(len(posterior_beta_x), len(posterior_beta_y))
bolfi_posterior_logpdf_default = (np.vectorize(lambda a, b: bolfi_posterior_ep.logpdf(np.array([a, b]))))(posterior_beta_x, posterior_beta_y)
bolfi_posterior_logpdf_customised = bolfi_posterior_ep.logpdf(np.vstack([posterior_beta_x.ravel(), posterior_beta_y.ravel()]).T).reshape(len(posterior_beta_x), len(posterior_beta_y))

bolfi_posterior_contour_plots, (bolfi_posterior_pdf_contour_plot,
                                bolfi_posterior_logpdf_contour_plot) = plt.subplots(1, 2, figsize = (15, 6))

bolfi_posterior_pdf_contour = bolfi_posterior_pdf_contour_plot.contourf(posterior_beta_x, posterior_beta_y, bolfi_posterior_pdf_default, levels = 30, cmap = "cividis", alpha = 0.7)
bolfi_posterior_pdf_contour_plot.set_title("BOLFI posterior pdf")
bolfi_posterior_pdf_contour_plot.set_xlabel(bolfi_posterior_ep.model.parameter_names[0])
bolfi_posterior_pdf_contour_plot.set_ylabel(bolfi_posterior_ep.model.parameter_names[1])
bolfi_posterior_pdf_contour_plot_cbar = bolfi_posterior_contour_plots.colorbar(bolfi_posterior_pdf_contour, ax = bolfi_posterior_pdf_contour_plot, label='density')

bolfi_posterior_logpdf_contour = bolfi_posterior_logpdf_contour_plot.contourf(posterior_beta_x, posterior_beta_y, bolfi_posterior_logpdf_customised, levels = 30, cmap = "cividis", alpha = 0.7)
bolfi_posterior_logpdf_contour_plot.set_title("BOLFI posterior log-pdf")
bolfi_posterior_logpdf_contour_plot.set_xlabel(bolfi_posterior_ep.model.parameter_names[0])
bolfi_posterior_logpdf_contour_plot.set_ylabel(bolfi_posterior_ep.model.parameter_names[1])
bolfi_posterior_logpdf_contour_plot_cbar = bolfi_posterior_contour_plots.colorbar(bolfi_posterior_logpdf_contour, ax = bolfi_posterior_logpdf_contour_plot, label='log-density')

plt.tight_layout()
plt.show()









