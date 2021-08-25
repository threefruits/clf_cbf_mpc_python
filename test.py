import rps.robotarium as robotarium
# from rps.utilities.transformations import *
# from rps.utilities.barrier_certificates import *
# from rps.utilities.misc import *
# from rps.utilities.controllers import *
from clf_cbf_nmpc import CLF_CBF_NMPC,Simple_Catcher
import numpy as np
import casadi as ca
from observer import Observer
from estimator import Estimator



# N = 2
N = 2

a = np.array([[-1.2, -1.2 , np.pi/6]]).T
# a = np.array([[0, 0 ,0]]).T
d = np.array([[1, 1, np.pi*0.75]]).T # 
initial_conditions = a
for idx in range(1, N):
#   d = np.array([[2.6*np.random.rand()-1.0, 2.6*np.random.rand()-1.0, 6.28*np.random.rand()-3.14]]).T
    initial_conditions = np.concatenate((initial_conditions, d), axis=1)
# print('Initial conditions:')
# print(initial_conditions)

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)


def is_done(all_states, goal):
    self_state = x[0]
    other_states = x[1:]

    # Check boundaries
    if(self_state[1]>3 or self_state[1]<-1.5 or self_state[0]>3 or self_state[0]<-1.5):
        print('Out of boundaries !!')
        return True
    # Reached goal?
    if abs(self_state[0]-goal[0])<0.01 and abs(self_state[1]-goal[1])<0.01 and abs(self_state[2]-goal[2])<0.005:
        print('Reach goal successfully!')
        return True

    for idx in range(np.size(other_states, 0)):
        # if(other_states[idx][0]>1.5 or other_states[idx][0]<-1.5 or other_states[idx][1]>1.5 or other_states[idx][1]<-1.5 ):
        #     print('Vehicle %d is out of boundaries !!' % idx+1)
        #     return True
        distSqr = (self_state[0]-other_states[idx][0])**2 + (self_state[1]-other_states[idx][1])**2
        if distSqr < (0.2)**2:
            print('Get caught, mission failed !')
            return True
    
    return False


x = r.get_poses().T
r.step()

i=0
times = 0

obsrvr = Observer(x, 0.1, 6)

mpc_horizon = 10
T = 0.1
m_cbf = 2
m_clf = 0
gamma_k = 0.2
alpha_k = 0.1
clf_cbf_nmpc_solver = CLF_CBF_NMPC(mpc_horizon, T, m_cbf, m_clf, gamma_k, alpha_k)

dist = 0.8
goal = np.array([x[1][0]-dist*np.cos(x[1][2]), x[1][1]-dist*np.sin(x[1][2]), x[1][2]])

while (is_done(x, goal)==False):
    print('\n----------------------------------------------------------')
    print("Iteration %d" % times)

    x = r.get_poses().T
    print('State: ', x[0])
    print('Goal:  ', goal)
    

    # Observe & Predict
    obsrvr.feed(x)
    f = lambda x_, u_: x_-x_ + u_
    # print(obsrvr.vel[1:])
    estmtr = Estimator(x[1:], obsrvr.vel[1:], f, 0.1, 10)
    estmtr.predict()
    # print(estmtr.predict_states)
    global_states_sol, controls_sol, local_states_sol = clf_cbf_nmpc_solver.solve(x[0], goal, np.concatenate((np.array([obsrvr.states[1:]]), estmtr.predict_states), axis = 0))
    # attacker_u = controls_sol[0]
    attacker_u = np.array([0, 0])

    # defender_u = Simple_Catcher(x[0],x[1])

    half_fov = 0.5* (np.pi* 1/2)    # 120 degrees of FoV
    agl = np.cos(half_fov)
    # control barrier function for FoV 
    h = lambda ori_, obj_: (ori_[0]*obj_[0]+ori_[1]*obj_[1])/ca.sqrt((obj_[0]**2+obj_[1]**2)) -np.cos(agl) 
    # for i in range(clf_cbf_nmpc_solver.M_CBF):
    i=0
    ori_i = [ca.cos(clf_cbf_nmpc_solver.opt_states[i, 2]), ca.sin(clf_cbf_nmpc_solver.opt_states[i, 2])]
    obj_i = [clf_cbf_nmpc_solver.goal_local[0]-clf_cbf_nmpc_solver.opt_states[i, 0], clf_cbf_nmpc_solver.goal_local[1]-clf_cbf_nmpc_solver.opt_states[i, 1]]
    print('ori = [%f , %f]' % (clf_cbf_nmpc_solver.opti.value(ori_i[0]), clf_cbf_nmpc_solver.opti.value(ori_i[1])))
    print('    = %f' % (clf_cbf_nmpc_solver.opti.value(ca.atan2(ori_i[0],ori_i[1]))/np.pi*180))
    print('obj = [%f , %f]'% (clf_cbf_nmpc_solver.opti.value(obj_i[0]), clf_cbf_nmpc_solver.opti.value(obj_i[1])))
    print('    = %f' % (clf_cbf_nmpc_solver.opti.value(ca.atan2(obj_i[0],obj_i[1]))/np.pi*180))
    print('h(i)=', clf_cbf_nmpc_solver.opti.value(h(ori_i, obj_i)))

    dxu = np.zeros([N,2])
    dxu[0] = np.array([attacker_u[0],attacker_u[1]])

    for idx in range(1, N):
        # defender_u = Simple_Catcher(x[0],x[idx])
        # dxu[idx] = defender_u
        # dxu[idx] = np.array([0, 0]) 
        dxu[idx] = np.array([0, 0]) 
    # for idx in range(3, N)
    #     defender_u = Simple_Catcher(x[0],x[idx])
    #     dxu[idx] = defender_u
    #     dxu[idx] = np.array([0.2, 0.02]) 

    r.set_velocities(np.arange(N), dxu.T)

    times+=1
    i+=1
    r.step()
    print('----------------------------------------------------------\n')

r.call_at_scripts_end()
