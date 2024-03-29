from directions import *
from euler import *
from gp import *
from reconstruction import *
from mesh import *
from traj_reader import *
import argparse, os

parser = argparse.ArgumentParser(description='SINATRA Pro')
args = parser.parse_args()

protA = "wt_fit"
protB = "perturbed"
n_sample = 100
sm_radius = 1.0 #r
n_cone = 20 #c
n_direction_per_cone = 8 #d
cap_radius = 0.8 #theta
n_filtration = 120 #l
ec_type = "DECT"
verbose = True
func = "linear"
parallel = True
label_type = "continuous"
from_pdb= False
selection = "protein"
by_frame = False


#for running on personal computer
"""
pdbpathA = '/Users/baihesun/Desktop/sp/sinatra_pro_lin_ser'
pdbpathB = '/Users/baihesun/Desktop/sp/perturbed'
directory = '/Users/baihesun/Desktop/sp/trial_1'
datapathA = '/Users/baihesun/Desktop/sp/sinatra_pro_lin_data_ser'
datapathB = '/Users/baihesun/Desktop/sp/sinatra_pro_lin_data_wt'

struct_file_A = '/Users/baihesun/Desktop/sp/sinatra_pro_lin_wt/wt_fit.pdb'
traj_file_A = struct_file_A
struct_file_B = '/Users/baihesun/Desktop/sp/sinatra_pro_perturbed/perturbed.pdb'
traj_file_B = struct_file_B
"""

#for running on oscar
#""""
pdbpathA = '/users/bsun14/sp/sinatra_pro_lin_wt'
pdbpathB = '/users/bsun14/sp/sinatra_pro_perturbed'
directory = '/users/bsun14/sp/trial_1'
datapathA = '/users/bsun14/sp/sinatra_pro_lin_data_wt'
datapathB = '/users/bsun14/sp/sinatra_pro_lin_data_ser'

struct_file_A = '/users/bsun14/sp/sinatra_pro_lin_wt/wt_fit.pdb'
traj_file_A = struct_file_A
struct_file_B = '/users/bsun14/sp/sinatra_pro_perturbed/perturbed.pdb'
traj_file_B = struct_file_B
#"""

""""
pdbpathA = '/users/bsun14/sp/sinatra_pro_lin_wt'
pdbpathB = '/users/bsun14/sp/sinatra_pro_lin_ser'
directory = '/users/bsun14/sp/sinatra_pro_lin_out'
datapathA = '/users/bsun14/sp/sinatra_pro_lin_data_wt'
datapathB = '/users/bsun14/sp/sinatra_pro_lin_data_ser'

struct_file_A = '/users/bsun14/sp/sinatra_pro_lin_wt/wt_fit.pdb'
traj_file_A = struct_file_A
struct_file_B = '/users/bsun14/sp/sinatra_pro_lin_ser/ser_fit.pdb'
traj_file_B = struct_file_B
"""

""""
#original data
struct_file_A='/users/bsun14/tem/WT.gro'
traj_file_A='/users/bsun14/tem/WT.xtc'
struct_file_B='/users/bsun14/tem/R164S.gro'
traj_file_B='/users/bsun14/tem/R164S.xtc'
"""

#directory = '/Users/baihesun/Desktop/sp/sinatra_pro_lin_out'

## Read trajectory file and output aligned protein structures in pdb format
if not from_pdb:
    convert_traj_pdb_aligned(protA, protB,
            struct_file_A=struct_file_A,
            traj_file_A=traj_file_A,
            struct_file_B=struct_file_B,
            traj_file_B=traj_file_B,
            align_frame=0,
            n_sample=n_sample,
            selection=selection,
            offset=0,
            directory=directory,
            single=True, ## single="True" is for single run purpose, "False" for duplicate runs purpose which groups and names file with the frame offset.
            verbose=verbose)

#####################
### IF you already have your own aligned structure, start here
#####################

if not from_pdb:
    directory_pdb_A = "%s/pdb/%s/"%(directory,protA)
    directory_pdb_B = "%s/pdb/%s/"%(directory,protB)
    reference_pdb_file = "%s/%s_frame0.pdb"%(directory_pdb_A,protA) ## which pdb to use for visualization

"""
directory = directory
directory_pdb_A = pdbpathA
directory_pdb_B = pdbpathB
directory_data_A = datapathA
directory_data_B = datapathB
"""

directory_data_A = datapathA
directory_data_B = datapathB


#_pdb_file = "%s/%s.pdb" % (directory_pdb_A, protA)

convert_pdb_mesh(protA,protB,
        n_sample=n_sample,
        sm_radius=sm_radius,
        directory_pdb_A=directory_pdb_A,
        directory_pdb_B=directory_pdb_B,
        directory_mesh="%s/msh/"%(directory),
        parallel=parallel,
        n_core=-1,
        verbose=verbose)

directions = generate_equidistributed_cones(n_cone=n_cone, n_direction_per_cone=n_direction_per_cone,
                                            cap_radius= cap_radius, hemisphere=False)

np.savetxt("%s/directions_%d_%d_%.2f.txt"%(directory,n_cone,n_direction_per_cone,cap_radius),directions)

X, y, not_vacuum = compute_ec_curve_folder(label_type = label_type, directory_data_A = directory_data_A, directory_data_B = directory_data_B,
                                           protA = protA ,protB= protB, directions = directions,
                                           n_sample=n_sample, ec_type="DECT",
                                           n_filtration=n_filtration, sm_radius=sm_radius,
                                           directory_mesh_A = "%s/msh/%s_%.1f"%(directory,protA,sm_radius),
                                           directory_mesh_B = "%s/msh/%s_%.1f"%(directory,protB,sm_radius),
                                           parallel=parallel, n_core=-1, verbose=True)

np.savetxt("%s/%s_%s_%s_%.1f_%d_%d_%.2f_%d_norm_all.txt"%(directory,ec_type,protA,protB,sm_radius,n_cone,n_direction_per_cone,cap_radius,n_filtration),X)
np.savetxt("%s/notvacuum_%s_%s_%s_%.1f_%d_%d_%.2f_%d_norm_all.txt"%(directory,ec_type,protA,protB,sm_radius,n_cone,n_direction_per_cone,cap_radius,n_filtration),not_vacuum)
np.savetxt('%s/%s_%s_label_all.txt'%(directory,protA,protB),y)

kld, rates, delta, eff_samp_size = calc_rate(X,y, func= func, bandwidth= 0.01, n_mcmc= 100000,low_rank=True, parallel=parallel,
                                             n_core=-1, verbose=verbose)

#CHANGE BELOW
rates = np.absolute(rates)

np.savetxt("%s/rate_%s_%s_%s_%.1f_%d_%d_%.2f_%d.txt"%(directory,ec_type,protA,protB,sm_radius,n_cone,n_direction_per_cone,cap_radius,n_filtration),rates)

vert_prob = reconstruct_on_multiple_mesh(protA,protB,directions, rates=rates, not_vacuum=not_vacuum, n_sample=n_sample,
                                         n_direction_per_cone=n_direction_per_cone, n_filtration=n_filtration,
                                         sm_radius=sm_radius, directory_mesh="%s/msh/%s_%.1f"%(directory,protA,sm_radius),
                                         parallel=parallel, n_core=-1, verbose=verbose)

np.savetxt("%s/rate_atom_%s_%s_%s_%.1f_%d_%d_%.2f_%d.txt"%(directory,ec_type,protA,protB,sm_radius,n_cone,n_direction_per_cone,cap_radius,n_filtration),vert_prob)

write_vert_prob_on_pdb(vert_prob, protA=protA, protB=protB, selection='protein', pdb_in_file=reference_pdb_file,
        pdb_out_file="%s/rate_atom_%s_%s_%s_%.1f_%d_%d_%.2f_%d_all.pdb"%(directory,ec_type,protA,protB,sm_radius,n_cone,n_direction_per_cone,cap_radius,n_filtration))


print("SINATRA Pro calculation completed.")

def main():
    print("main")

if __name__ == "__main__":
    main()
    print("done")
