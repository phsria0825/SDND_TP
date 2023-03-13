# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        """
                             / d(x_1,z_1)  d(x_1,z_2)  ....  d(x_1,z_M) \ 
                             |     :         :                    :     |
        association_matrix = |     :         :                    :     |
                             \ d(x_N,z_1)  d(x_N,z_2)  ....  d(x_N,z_M) /

        N : tracks
        M : measurements

        알고리즘:
        1. 1번째 for문으로 행에 접근
        2. 2번째 for문으로 열에 접근
        3. track_list와 meas_list 배열에 있는 원소에 접근하여 MHD 계산
        4. MHD 계산결과를 association_matrix 변수에 저장
        """

        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))
        
        # initialize association matrix
        self.association_matrix = np.inf*np.ones((N,M)) 

        for i in range(N):
            for j in range(M):

                dist = self.MHD(track_list[i], meas_list[j], KF)
                if self.gating(dist, meas_list[j].sensor):
                    self.association_matrix[i,j] = dist

        self.unassigned_tracks = list(range(len(track_list)))
        self.unassigned_meas = list(range(len(meas_list)))
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # find closest track and measurement for next update
        # MHD의 최소거리가 inf면 이상치 데이터이기 때문에 nan처리
        if np.min(self.association_matrix) == np.inf:
            return np.nan, np.nan
        
        # get indices of minimum entry
        # MHD의 최소거리인 인덱스를 추출
        ij_min = np.unravel_index(np.argmin(self.association_matrix, axis=None),
                                  self.association_matrix.shape) 
        ind_track = ij_min[0]
        ind_meas = ij_min[1]

        # delete row and column for next update
        # 추출된 위치의 해당하는 행과 열 삭제후 다음 association_matrix 생성
        self.association_matrix = np.delete(self.association_matrix, ind_track, 0) 
        self.association_matrix = np.delete(self.association_matrix, ind_meas, 1)
        self.association_matrix = self.association_matrix

        # update this track with this measurement
        update_track = self.unassigned_tracks[ind_track] 
        update_meas = self.unassigned_meas[ind_meas]

        # remove this track and measurement from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        """
        d(x,z) <= chi2.ppf(p|df)
        p  : probability
        df : degree of freedom (which equals the dimension of the measurement space)
        """
        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        if MHD <= limit:
            return True
        else:
            return False
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        """
        Mahalanobis distance:
        d(x,z) = gamma.transpose() * S.inverse() * gamma
        
        gamma = z - H * x
        """

        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        H = meas.sensor.get_H(track.x)
        S_inv = np.linalg.inv(H * track.P * H.T + meas.R)
        gamma = KF.gamma(track, meas)
        
        return gamma.T*S_inv*gamma
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)
