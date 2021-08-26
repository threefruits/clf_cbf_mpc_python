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


def is_done(all_states, goal):
    self_state = x[0]
    other_states = x[1:]

    # Check boundaries
    if(self_state[1]>5 or self_state[1]<-5 or self_state[0]>5 or self_state[0]<-5):
        print('Out of boundaries !!')
        return True
    # Reached goal?
    if abs(self_state[0]-goal[0])<0.01 and abs(self_state[1]-goal[1])<0.01 and abs(self_state[2]-goal[2])<0.05:
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

#######################################################################
########################### __main__ begins ###########################
#######################################################################


# N = 2
N = 5

a = np.array([[-1.4, -1.4 , np.pi *1/8 ]]).T
# a = np.array([[0, 0 ,0]]).T
d = np.array([[0.8, 0.8, np.pi * 3.1/3]]).T # 
initial_conditions = a
initial_conditions = np.concatenate((initial_conditions, d), axis=1)
# for idx in range(1, N):
for idx in range(2, N):
    d = np.array([[2.6*np.random.rand()-1.0, 2.6*np.random.rand()-1.0, 6.28*np.random.rand()-3.14]]).T
    initial_conditions = np.concatenate((initial_conditions, d), axis=1)
# print('Initial conditions:')
# print(initial_conditions)

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

x = r.get_poses().T
r.step()
i=0
times = 0

obsrvr = Observer(x, 0.1, 6)

mpc_horizon = 30
T = 0.1
m_cbf = 5
m_fov = 10
m_clf = 0
gamma_k = 0.2
gamma_fov = 0.3
alpha_k = 0.01
Q = np.array([[1.5, 0, 0.0],[0, 1.5, 0.0],[0.0, 0.0, 0.005]])
R = np.array([[0.01, 0.0], [0.0, 0.0001]])
W = np.array([[10, 0.0], [0.0, 200]])
clf_cbf_nmpc_solver = CLF_CBF_NMPC(mpc_horizon, T,Q,R,W, m_cbf, m_fov, m_clf, gamma_k, gamma_fov, alpha_k)

dist = 0.5
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
    global_states_sol, controls_sol, local_states_sol, slack_sol = clf_cbf_nmpc_solver.solve(x[0], goal, x[1], np.concatenate((np.array([obsrvr.states[1:]]), estmtr.predict_states), axis = 0))
    attacker_u = controls_sol[0]
    # attacker_u = np.array([0, 0])

    print('fov_slack =', slack_sol[0])
    print('clf_slack =', slack_sol[1])
    
    dxu = np.zeros([N,2])
    dxu[0] = np.array([attacker_u[0],attacker_u[1]])

    dxu[1] = np.array([0, 0])
    for idx in range(2, N):
        # defender_u = Simple_Catcher(x[0],x[idx])
        # dxu[idx] = defender_u
        dxu[idx] = np.array([0.1, 0.15]) 


    r.set_velocities(np.arange(N), dxu.T)

    times+=1
    i+=1
    r.step()
    print('----------------------------------------------------------\n')

r.call_at_scripts_end()
