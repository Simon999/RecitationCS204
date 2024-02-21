import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling, GibbsSampling
from pgmpy.inference import VariableElimination

# Define the Bayesian Network structure
model = BayesianNetwork([('Rain', 'Sprinkler'), ('Rain', 'GrassWet'), ('Sprinkler', 'GrassWet')])

# Define the Conditional Probability Distributions (CPDs)
cpd_rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.8], [0.2]])
cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2, values=[[0.9, 0.5], [0.1, 0.5]],
                           evidence=['Rain'], evidence_card=[2])
cpd_grass_wet = TabularCPD(variable='GrassWet', variable_card=2, values=[[1.0, 0.1, 0.1, 0.01], [0.0, 0.9, 0.9, 0.99]],
                           evidence=['Rain', 'Sprinkler'], evidence_card=[2, 2])

# Add CPDs to the model
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_grass_wet)

# Check model validity
assert model.check_model()

# Prior Sampling
sampler = BayesianModelSampling(model)
prior_samples = sampler.forward_sample(size=1000)

# Rejection Sampling
rejection_samples = sampler.rejection_sample(evidence={'GrassWet': 1}, size=1000)

# Gibbs Sampling
gibbs_sampler = GibbsSampling(model)
gibbs_samples = gibbs_sampler.sample(size=1000, evidence={'GrassWet': 1})

prior_samples.head(), rejection_samples.head(), gibbs_samples.head()
