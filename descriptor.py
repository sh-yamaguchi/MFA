#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import lib.pymfa as pymfa

setting = np.loadtxt("data/setting.txt",str)
mean_plane = int(setting[0,1]);  origin = int(setting[1,1]); x_axis = int(setting[2,1]); n = list(setting[3,1]); n = [int(i) for i in n]
grid_x = int(setting[4,1]); grid_y = int(setting[5,1]); grid_z = int(setting[6,1])
threshold = float(setting[7,1])

ex = pymfa.MolecularFieldAnalysis(mean_plane, origin, x_axis, n, grid_x, grid_y, grid_z, threshold)
mol_name = np.loadtxt("data/mol_name.txt",str)

descriptor = ex.indicator_field(mol_name[range(24)])
np.savetxt("descriptor.txt",descriptor[1])
