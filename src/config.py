"""
File: config.py
===============
This file holds some package-wide constant variables
"""
import numpy as np

STARTING_CONCS = [0.25, 0.25, 0]

OUTPUT_FOLDER = "./simulations"
DEFAULT_INIT = "random"

DEFAULT_TEMP_LOWERBOUND = 273
DEFAULT_TEMP_UPPERBOUND = 473

DEFAULT_LAPLACE_MATRIX = np.array([[0.05, 0.2, 0.05],
                                   [0.2, -1.0, 0.2],
                                   [0.05, 0.2, 0.05]], np.float64)

RESTRICTED_LAPLACE_MATRIX = np.array([[0.3, 0.08, 0.04],
                                      [0.08, -1.0, 0.08],
                                      [0.04, 0.08, 0.3]], np.float64)
