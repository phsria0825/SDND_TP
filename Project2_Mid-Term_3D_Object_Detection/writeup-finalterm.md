# Writeup: Sensor Fusion and Object Tracking (Finalterm)

## Step-1: Implement an EKF to track

you will implement an EKF to track a single real-world target with lidar measurement input over time

* The single track is already initialized for you, so don't worry about track initialization right now.

* In `student/filter.py`, implement the `predict()` function for an EKF. Implement the `F()` and `Q()` functions to calculate a system matrix for constant velocity process model in 3D and the corresponding process noise covariance depending on the current timestep dt. Note that in our case, dt is fixed and you should load it from `misc/params.py`. However, in general, the timestep might vary. At the end of the prediction step, save the resulting x and P by calling the functions `set_x()` and `set_P()` that are already implemented in `student/trackmanagement.py.`

* Implement the `update()` function as well as the `gamma()` and `S()` functions for residual and residual covariance. You should call the functions `get_hx` and `get_H` that are already implemented in `students/measurements.py` to get the measurement function evaluated at the current state, h(x), and the Jacobian H. Note that we have a linear measurement model for lidar, so h(x)=H*x for now. You should use h(x) nevertheless for the residual to have an EKF ready for the nonlinear camera measurement model you'll need in Step 4. Again, at the end of the update step, save the resulting x and P by calling the functions `set_x()` and set_`P()` that are already implemented in `student/trackmanagement.py`.

* Use `numpy.matrix()` for all matrices as learned in the exercises.

The project can be run by running
```
python loop_over_dataset.py
```

The changes are made in `loop_over_dataset.py`
```python
## Select Waymo Open Dataset file and frame numbers
data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
show_only_frames = [150, 200] # show only frames in interval for debugging
sequence = "2"

## Initialize object detection
configs_det = det.load_configs(model_name='fpn_resnet') # options are 'darknet', 'fpn_resnet'
model_det = det.create_model(configs_det)

# Midterm, Finalterm의 results_fullpath를 결정해주는 flag (True : Midterm, False : Finalterm)
path_flag = False

## Uncomment this setting to restrict the y-range in the Final project
## comment this setting to restrict the y-range in the Mid project
configs_det.lim_y = [-5, 10] 

## Selective execution and visualization
exec_detection = []
exec_tracking = ['perform_tracking']
exec_visualization = ['show_tracks']
```

Changes are made in `filter.py`

**`F(self)`**
This function is constant velocity process model for tracking

```python
F = np.identity(self.dim_state)
        F[0,3] = self.dt
        F[1,4] = self.dt
        F[2,5] = self.dt

        return np.matrix(F)
```

**`Q(self)`**
This function is constant velocity process noise covariance for tracking

```python
q3 = (self.dt**3)*self.q / 3
        q2 = (self.dt**2)*self.q / 2
        q1 = self.dt*self.q

        return np.matrix([[q3,  0,   0,   q2,  0,   0],
                          [0,   q3,  0,   0,   q2,  0],
                          [0,   0,   q3,  0,   0,   q2],
                          [q2,  0,   0,   q1,  0,   0],
                          [0,   q2,  0,   0,   q1,  0],
                          [0,   0,   q2,  0,   0,   q1]])
```

**`predict(self, track)`**
predict state x and estimation error covariance P to next timestep, save x and P in track

```python
x = self.F() * track.x
        P = self.F() * track.P * self.F().transpose() + self.Q()

        track.set_x(x)
        track.set_P(P)
```

**`update(self, track, meas)`**
update state x and covariance P with associated measurement, save x and P in track

```python
H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)
        K = track.P * H.transpose() * np.linalg.inv(S)

        x = track.x + K * gamma
        P = (np.identity(self.dim_state) - K * H) * track.P

        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)
```

**`gamma(self, track, meas), S(self, track, meas, H)`**
```python
def gamma(self, track, meas):
        gamma_ = meas.z - meas.sensor.get_hx(track.x)
        return gamma_


    def S(self, track, meas, H):
        S_ = H * track.P * H.transpose() + meas.R
        return S_
```



















Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?


### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?


### 4. Can you think of ways to improve your tracking results in the future?