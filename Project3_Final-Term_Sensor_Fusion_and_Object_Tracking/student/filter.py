# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state # process model dimension
        self.dt = params.dt # time increment
        self.q = params.q # process noise variable for Kalman filter Q

    def F(self):
        """
        F is motion model matrix for prediction

        Linear motion model:
        px' = px + vx * dt
        py' = py + vy * dt
        pz' = pz + vz * dt
        vx' = vx
        vy' = vy
        vz' = vz

        F = [1  0   0   dt  0   0]
            [0  1   0   0   dt  0]
            [0  0   1   0   0  dt]
            [0  0   0   1   0   0]
            [0  0   0   0   1   0]
            [0  0   0   0   0   1]
        """
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        F = np.identity(self.dim_state)
        F[0,3] = self.dt
        F[1,4] = self.dt
        F[2,5] = self.dt

        return np.matrix(F)
        
        ############
        # END student code
        ############ 

    def Q(self):
        """
        Process Noise Covariance Q

        Q = [0 0 0 0 0 0]
            [0 0 0 0 0 0]
            [0 0 0 0 0 0]
            [0 0 0 q 0 0]
            [0 0 0 0 q 0]
            [0 0 0 0 0 q]

        Let, F = [1  0   0   t   0   0]
                 [0  1   0   0   t   0]
                 [0  0   1   0   0   t]
                 [0  0   0   1   0   0]
                 [0  0   0   0   1   0]
                 [0  0   0   0   0   1]

                / dt                               [q3  0   0   q2  0   0 ]
        Q(dt) = |                                  [0   q3  0   0   q2  0 ]
                |   (F * Q * F.transpose()) dt  =  [0   0   q3  0   0   q2]
                |                                  [q2  0   0   q1  0   0 ]
                / 0                                [0   q2  0   0   q1  0 ]
                                                   [0   0   q2  0   0   q1 ]

        q3 = (dt**3)*q / 3
        q2 = (dt**2)*q / 2
        q1 = dt*q
        
        """
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        q3 = (self.dt**3)*self.q / 3
        q2 = (self.dt**2)*self.q / 2
        q1 = self.dt*self.q

        return np.matrix([[q3,  0,   0,   q2,  0,   0],
                          [0,   q3,  0,   0,   q2,  0],
                          [0,   0,   q3,  0,   0,   q2],
                          [q2,  0,   0,   q1,  0,   0],
                          [0,   q2,  0,   0,   q1,  0],
                          [0,   0,   q2,  0,   0,   q1]])
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        """
        x_ = F * x
        P_ = F * P * F.transpose() + Q
        """
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        x = self.F() * track.x
        P = self.F() * track.P * self.F().transpose() + self.Q()

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############ 

    def update(self, track, meas):
        """
        gamma = z - H * x
        S = H * P_ * H.transpose() + R
        K = P_ * H.transpose() * S.inverse()
        x = x_ + K * gamma
        P = (I - K * H) * P_
        """
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############

        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)
        K = track.P * H.transpose() * np.linalg.inv(S)

        x = track.x + K * gamma
        P = (np.identity(self.dim_state) - K * H) * track.P

        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)

        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        gamma_ = meas.z - meas.sensor.get_hx(track.x)
        return gamma_
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        S_ = H * track.P * H.transpose() + meas.R
        return S_
        
        ############
        # END student code
        ############ 