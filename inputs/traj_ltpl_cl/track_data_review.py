# Track Data Review
# 该数据格式可由global_trajectory计算得到
#------------------------
#LTPL Trajectory
#The output csv contains the source information of the global race trajectory and map information via the normal vectors. The array is of size [no_points x 12] where no_points depends on step size and track length. The seven columns are structured as follows:
#x_ref_m: float32, meter. X-coordinate of reference line point (e.g. center line of the track).
#y_ref_m: float32, meter. Y-coordinate of reference line point (e.g. center line of the track).
#width_right_m: float32, meter. Distance between reference line point and right track bound (along normal vector).
#width_left_m: float32, meter. Distance between reference line point and left track bound (along normal vector).
#x_normvec_m: float32, meter. X-coordinate of the normalized normal vector based on the reference line point.
#y_normvec_m: float32, meter. Y-coordinate of the normalized normal vector based on the reference line point.
#alpha_m: float32, meter. Solution of the opt. problem holding the lateral shift in m for the reference line point.
#s_racetraj_m: float32, meter. Curvi-linear distance along the race line.
#psi_racetraj_rad: float32, rad. Heading of raceline in current point from -pi to +pi rad. Zero is north.
#kappa_racetraj_radpm: float32, rad/meter. Curvature of raceline in current point.
#vx_racetraj_mps: float32, meter/second. Target velocity in current point.
#ax_racetraj_mps2: float32, meter/second². Target acceleration in current point. We assume this acceleration to be constant from current point until next point.

import numpy as np
import csv
import math 
import matplotlib.pyplot as plt

index = 0
with open('traj_ltpl_cl_berlin_display.csv', mode='r',encoding='utf-8',newline='') as f:
    reader = csv.reader(f)
    x_ref = []
    y_ref = []
    w_r = []
    w_l  = []
    x_v = []
    y_v = []
    alpha = []
    s_m = []
    psi = []
    kappa=[]
    v = []
    a = []
    for row in reader:
        if index > 0 and row[0][0]!='#':
            x_ref.append(float(row[0]))
            y_ref.append(float(row[1])) 
            w_r.append(float(row[2])) 
            w_l.append(float(row[3]))
            x_v.append(float(row[4]))
            y_v.append(float(row[5]))
            alpha.append(float(row[6]))
            s_m.append(float(row[7]))
            psi.append(float(row[8]))
            kappa.append(float(row[9]))
            v.append(float(row[10]))
            a.append(float(row[11]))
        index = index+1

#print(len(x_ref))
data = np.vstack((x_ref,y_ref,w_r,w_l,x_v,y_v,alpha,s_m,psi,kappa,v,a))

#print(x_ref)
plt.figure(1)
plt.plot(x_ref,y_ref)
plt.axis('equal')
plt.title('Input Track Review')

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(s_m,a)
plt.ylabel('acc')
plt.subplot(2,1,2)
plt.plot(s_m,v)
plt.ylabel('Spd[m/s]')
plt.xlabel('Distance[m]')

#######################
##基于alpha的raceline重构
#alpha的定义参见Minimum curvature trajectory planning and control for an autonomous race car section 3.4.1
#alpha用于在法向量方向对最佳点进行移动，其范围为【-w_l+w_veh/2, +w_l-w_veh/s】
######################
# 重构boundary与raceline

bond_up_x = []
bond_up_y = []
bond_down_x = []
bond_down_y = []
raceline_x = []
raceline_y = []
for i in range(len(alpha)): 
    bond_up_x.append(x_ref[i] + x_v[i]*-1*w_l[i])
    bond_up_y.append(y_ref[i] + y_v[i]*-1*w_l[i])
    bond_down_x.append(x_ref[i] + x_v[i]*+1*w_r[i])
    bond_down_y.append(y_ref[i] + y_v[i]*+1*w_r[i])
    raceline_x.append(x_v[i]*alpha[i] + x_ref[i])
    raceline_y.append(y_v[i]*alpha[i] + y_ref[i])
# input文件中的ref_track是重计算的结果，已经不是中心线了。所以结果中ref_track与raceline差异很小。


plt.figure(1)
plt.plot(bond_up_x,bond_up_y)
plt.plot(bond_down_x,bond_down_y)
plt.plot(raceline_x,raceline_y)
plt.legend(['ref_track','Up Bound','Low Bound','Opt Res'])
plt.show()
