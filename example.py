import numpy as np
import jtsocperovskite as gr

# With this code you can calculate the non-gyrotropic (rhomr) and the gyrotropic (rhogy) signals
# The parameters are tha same used for simulations on the paper

cut = (0,1,1)
f = np.linspace(1.77,3.10,704)
solver = gr.Solver(delta=1.188, cut=cut, FE=0.45, FT=0.13, GE=0.02, xiSO=0.02, dump=0.18, CT=4, tpd=1.2, ssd=1, sdd=0.82, pdd=0.29, ddd=0.07, spin_dir=(0,0,1))
rhomr, rhogy = solver.solve(f)


# Data save
np.savetxt("../results/ngsignal.out", np.array([f, rhomr]).T, fmt="%.8e", delimiter="\t") # Non-gyrotropic signal
np.savetxt("../results/gysignal.out", np.array([f, rhogy]).T, fmt="%.8e", delimiter="\t") # Gyrotropic signal