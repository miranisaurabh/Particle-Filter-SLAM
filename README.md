particle_filter.py

- Loads the provided data
- Has a sympy and lambda function based transformation matrix for faster computations. It creates a function which takes all the joint angles, current pose of the robot & LIDAR scans lin sensor frame and returns the xyz coordinates in world frame
- Initializes all the parameters required (MAP,correlation etc)
- Now inside the loop:
- Takes the lidar scan, gets joint angles using timestamp matching
- If it is the very 1st timestamp then assumes the robot position to be (0,0,0) and updates the log odds map accordingly
- If not then gets the positions of particles using motion model and noise
- Then transforms the lidar scan to world frame with respect to each particle
- Gets the correlation value using mapCorrelation() function
- Updates the weights and finds the best particle
- Gets the free space using the best particlle's position and the LIDAR scan that this particle observes
- Log odds map is updated using this particles position and the LIDAR scan that this particle observes
- Overfitting is done, i.e. limits on the maximum and minimum values in log odds map

The code is also well commented for more details.
-------------------------------------------------------------------------------------------------------
texture.py

- It loads all the provided data and the data saved from particle filter (MAP and trajectory)
- Has a sympy and lambda function based transformation matrix for faster computations. It creates a function which takes all the joint angles, current pose of the robot & LIDAR scans lin sensor frame and returns the xyz coordinates in world frame
- Calculates the xyz coordinates in optical frame of IR camera from the depth image
- Transforms these coordinates to world frame 
- Thresholds z to find only the ground plane
- Finds the corressponding xyz coordinates in optical frame of IR camera
- These ground xyz coordinates in optical frame are transformed to optical frame of RGB using extrinsic values
- Finds the corresponding pixels in RGB frame, check if they are feasible, i.e lie within RGB image dimensions
- The pixels which are feasible, their corresponding colors are then matched with the corresponding xyz coordinates in the world frame (we have these after thresholding z for ground)
- xy are converted to grid cells and the map is now filled with corresponding colors.

The code is also well commented for more details.

