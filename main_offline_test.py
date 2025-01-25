import numpy as np
import configparser
import pickle
import os.path as osfuncs
import hashlib  #摘要算法库
import logging
import matplotlib
#matplotlib.use('TkAgg') # For qt problem
import matplotlib.pyplot as plt


# custom modules
import graph_ltpl

# 生成一个伪MD5码
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

track_path = "inputs/traj_ltpl_cl/traj_ltpl_cl_berlin.csv"
#calculated_md5 = md5('inputs/traj_ltpl_cl/traj_ltpl_cl_shanghai.csv') # 文件MD5码
calculated_md5 = md5(track_path) # 文件MD5码

# ------------------------------------------------------------------------------------------------------------------
# SETUP GRAPH ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# load data from csv files
refline, t_width_right, t_width_left, normvec_normalized, alpha, length_rl, vel_rl, kappa_rl \
    = graph_ltpl.imp_global_traj.src.import_globtraj_csv.import_globtraj_csv(import_path=track_path)
# load graph configuration
graph_config = configparser.ConfigParser()
if not graph_config.read('params/ltpl_config_offline.ini'):
    raise ValueError('Specified graph config file does not exist or is empty!')

# calculate closed race line parameters
# s, x, y, kappa, vel
s = np.concatenate(([0], np.cumsum(length_rl)))
xy = refline + normvec_normalized * alpha[:, np.newaxis]
raceline_params = np.column_stack((xy, kappa_rl, vel_rl))

# determine if track is closed or unclosed (check if end and start-point are close together)
closed = (np.hypot(xy[0, 0] - xy[-1, 0], xy[0, 1] - xy[-1, 1])
            < graph_config.getfloat('LATTICE', 'closure_detection_dist'))
if closed:
    logging.getLogger("local_trajectory_logger").debug("Input line is interpreted as closed track!")

    # close line
    glob_rl = np.column_stack((s, np.vstack((raceline_params, raceline_params[0, :])))) #强行闭环
else:
    logging.getLogger("local_trajectory_logger").debug("Input line is interpreted as _unclosed_ track!")
    glob_rl = np.column_stack((s[:-1], raceline_params)) # 开环数据

# based on curvature get index array for selection of normal vectors and corresponding raceline parameters 
# 通过曲率信息，对赛道上不同的地方进行变步长采样。曲率越大则采样约密

idx_array = graph_ltpl.imp_global_traj.src.variable_step_size. \
    variable_step_size(kappa=kappa_rl,
                        dist=length_rl,
                        d_curve=10.0,
                        d_straight=30.0,
                        curve_th=graph_config.getfloat('LATTICE', 'curve_thr'),
                        force_last=not closed)

#print('变步长重采样点')
#print(idx_array)

# extract values at determined positions
refline = refline[idx_array, :]
t_width_right = t_width_right[idx_array]
t_width_left = t_width_left[idx_array]
normvec_normalized = normvec_normalized[idx_array]
alpha = alpha[idx_array]
vel_rl = vel_rl[idx_array]
s_raceline = s[idx_array]

length_rl_tmp = []
for idx_from, idx_to in zip(idx_array[:-1], idx_array[1:]):
    length_rl_tmp.append(np.sum(length_rl[idx_from:idx_to]))

length_rl_tmp.append(0.0)
length_rl = list(length_rl_tmp)
# raceline的段落长
# print(length_rl)

# init graph base object
graph_base = graph_ltpl.data_objects.GraphBase.\
    GraphBase(lat_offset=graph_config.getfloat('LATTICE', 'lat_offset'),
                num_layers=np.size(alpha, axis=0),
                refline=refline,
                normvec_normalized=normvec_normalized,
                track_width_right=t_width_right,
                track_width_left=t_width_left,
                alpha=alpha,
                vel_raceline=vel_rl,
                s_raceline=s_raceline,
                lat_resolution=graph_config.getfloat('LATTICE', 'lat_resolution'),
                sampled_resolution=graph_config.getfloat('SAMPLING', 'stepsize_approx'),
                vel_decrease_lat=graph_config.getfloat('PLANNINGTARGET', 'vel_decrease_lat'),
                veh_width=graph_config.getfloat('VEHICLE', 'veh_width'),
                veh_length=graph_config.getfloat('VEHICLE', 'veh_length'),
                veh_turn=graph_config.getfloat('VEHICLE', 'veh_turn'),
                md5_params=calculated_md5,
                graph_id= 'test_graph',
                glob_rl=glob_rl,
                virt_goal_node=graph_config.getboolean('LATTICE', 'virt_goal_n'),
                virt_goal_node_cost=graph_config.getfloat('COST', 'w_virt_goal'),
                min_plan_horizon=graph_config.getfloat('PLANNINGTARGET', 'min_plan_horizon'),
                plan_horizon_mode=graph_config.get('PLANNINGTARGET', 'plan_horizon_mode'),
                closed=closed)


# set up state space
state_pos = graph_ltpl.offline_graph.src.gen_node_skeleton. \
    gen_node_skeleton(graph_base=graph_base,
                        length_raceline=length_rl,
                        var_heading=graph_config.getboolean('LATTICE', 'variable_heading'))


#对每个state_pos, 包含数个[temp_pos, temp_psi]
plt.figure(1)
num_point_long = len(state_pos)
num_point_width = len(state_pos[0][0]) 
print('long is '+str(num_point_long)+' blocks and width is '+str(num_point_width)+' points')
frame_left_x = []
frame_left_y = []
frame_right_x = []
frame_right_y = []
for i in state_pos:
    pos = i[0]
    psi = i[1]
    for point in pos:
        plt.plot(point[0],point[1],'.')
    for i in range(len(pos)):
        frame_left_x.append(pos[0][0])
        frame_left_y.append(pos[0][1])
        frame_right_x.append(pos[-1][0])
        frame_right_y.append(pos[-1][1])
plt.plot(frame_left_x,frame_left_y)
plt.plot(frame_right_x,frame_right_y)
plt.axis('equal')
plt.title('Skeleton of the track')
plt.show()

# convert to array of arrays
state_pos_arr = np.empty(shape=(len(state_pos), 2), dtype=object)
state_pos_arr[:] = state_pos

# generate edges (polynomials and coordinate arrays)
edge = graph_ltpl.offline_graph.src.gen_edges.gen_edges(state_pos=state_pos_arr,
                                                    graph_base=graph_base,
                                                    stepsize_approx=graph_config.getfloat('SAMPLING',
                                                                                        'stepsize_approx'),
                                                    min_vel_race=graph_config.getfloat('LATTICE', 'min_vel_race'),
                                                    closed=closed)