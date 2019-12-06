import config

#########################################
#       Gray-Scott Simulations          #
#########################################
base_gray_scott = {
    'n' : 2,
    'order' : -1,
    'diffusions' : [1, 0.5],
    'feed' : 0.0362,
    'kill' : 0.062,
    'temp' : None,
    'init' : "pointMass",
    'activationEnergies' : None,
    'startingConcs' : [0.25, 0.25, 0],
    'laplace_matrix' : config.DEFAULT_LAPLACE_MATRIX,
    'normalize_values' : True,
    'updates_per_frame' : 100,
}

default = {
    'feed' : 0.0362,
    'kill' : 0.062
}

circles = {
    'feed' : 0.03,
    'kill' : 0.0545
}

def get_gray_scott_config(param):
    out = base_gray_scott.copy()
    out["feed"] = param["feed"]
    out["kill"] = param["kill"]
    return out


#########################################
#         Reaction Simulations          #
#########################################

first_order = {
    'n' : 3,
    'order' : 1,
    'diffusions' : [1, 1, 0.5],
    'feed' : None,
    'kill' : None,
    'temp' : 350,
    'init' : "even",
    'activationEnergies' : [4, 4],
    'startingConcs' : [0.5, 0.25, 0],
    'laplace_matrix' : config.DEFAULT_LAPLACE_MATRIX,
    'normalize_values' : True,
    'updates_per_frame' : 1,
}

second_order = {
    'n': 3,
    'order': 2,
    'diffusions': [1, 1, 0.5],
    'feed': None,
    'kill': None,
    'temp': 298,
    'init': "even",
    'activationEnergies': [3, 3],
    'startingConcs': [0.5, 0.25, 0],
    'laplace_matrix': config.DEFAULT_LAPLACE_MATRIX,
    'normalize_values': True,
    'updates_per_frame': 1,
}

cellular_open = {
    'n': 3,
    'order': 2,
    'diffusions': [1, 1, 0.5],
    'feed': None,
    'kill': None,
    'temp': 350,
    'init': "cellular-open",
    'activationEnergies': [3, 3],
    'startingConcs': [0.5, 1, 0],
    'laplace_matrix': config.DEFAULT_LAPLACE_MATRIX,
    'normalize_values': True,
    'updates_per_frame': 1,
}

cellular_restricted = {
    'n': 3,
    'order': 2,
    'diffusions': [1, 1, 0.5],
    'feed': None,
    'kill': None,
    'temp': 350,
    'init': "cellular-open",
    'activationEnergies': [3, 3],
    'startingConcs': [0.5, 1, 0],
    'laplace_matrix': config.RESTRICTED_LAPLACE_MATRIX,
    'normalize_values': True,
    'updates_per_frame': 1,
}

