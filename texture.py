import load_data as data
import numpy as np
import matplotlib.pyplot as plt
import time
import sympy
from sympy import symbols, pprint
from sympy import sin, cos
from sympy import Matrix
from sympy.utilities.lambdify import lambdify
import cv2

# Get camera calibration
exIR_RGB = data.getExtrinsics_IR_RGB()
IRCalib = data.getIRCalib()
RGBCalib = data.getRGBCalib()

# Function for converting calibration values to matrix
def get_intrinsic(camera):
   
   K = np.array([[camera['fc'][0],0,camera['cc'][0]], [0,camera['fc'][1],camera['cc'][1]], [0,0,1]])
   return K

# Load Data
j0 = data.get_joint("joint/train_joint4")
l0 = data.get_lidar("lidar/train_lidar4")
r0 = data.get_rgb("cam/RGB_4")
d0 = data.get_depth("cam/DEPTH_4")
joint_TimeStamps = j0['ts']
joint_Angles = j0['head_angles']

# Load save data from Particle filter
current_positions = np.load('Positions4_texture.npy')

# Convert lidart timestamps from list of dictionary to array for easy computations
lidar_TimeStamps = np.zeros((1,np.size(l0)))
for count_lidar in range(np.size(l0)):
    lidar_TimeStamps[0][count_lidar] = l0[count_lidar]['t']

# Sympy implementation for Transformation matrices
(head_angle, 
 neck_angle,
 x, 
 y,
 pose_angle,
 lx,
 ly) = symbols(""" theta1 
                         theta2
                         x 
                         y
                         theta3
                         l_x
                         l_y""" , real = True)

T1 = Matrix([[cos(head_angle), 0, sin(head_angle),0],
    [0,1,0,0],
    [-sin(head_angle),0,cos(head_angle),0],
    [0,0,0,1]])

T2 = Matrix([[cos(neck_angle),-sin(neck_angle), 0,0],
    [sin(neck_angle),cos(neck_angle),0,0],
    [0,0,1,0],
    [0,0,0,1]])

T3 = Matrix([[1,0,0,0],
    [0,1,0,0],
    [0,0,1,0.4],
    [0,0,0,1]])

T_ltoR = T3*T2*T1

T4 = Matrix([[cos(pose_angle),-sin(pose_angle), 0,0],
    [sin(pose_angle),cos(pose_angle),0,0],
    [0,0,1,0],
    [0,0,0,1]])

T5 = Matrix([[1,0,0,x],
    [0,1,0,y],
    [0,0,1,0.93],
    [0,0,0,1]])
T_rtoW = T5*T4

T_full = T_rtoW*T_ltoR
T_full = sympy.simplify(T_full)
get_cam_world_pose = lambdify((head_angle,neck_angle, x, y, pose_angle),T_full)

# Optical regular matrix transformation (Inverse is required)
oRr = np.array([[0,-1,0,0],
                [0,0,-1,0],
                [1,0,0,0],
                [0,0,0,1]])
oRr_inv = np.linalg.inv(oRr)

# Get the intrinsics values
RGB_intrinsics = get_intrinsic(RGBCalib)
IR_intrinsics = get_intrinsic(IRCalib)
IR_inv = np.linalg.inv(IR_intrinsics)

# Array of 1's for converting to homogenous coordinates
ones_array = np.ones((1,512*424))

# Convert given extrinsic values to matrix (4x4)
IR_RGB = np.hstack((exIR_RGB['rgb_R_ir'],exIR_RGB['rgb_T_ir'].reshape(-1,1)))
IR_RGB = np.vstack((IR_RGB,np.array([0,0,0,1])))
IR_RGB_inv = np.linalg.inv(IR_RGB)

# Initialize MAP parameters (required when converting to grid)
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -17  #meters
MAP['ymin']  = -40
MAP['xmax']  =  33
MAP['ymax']  =  25 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

# create array of (u,v) to get the coordinates in sensor frame (v,u) order is used
grid_uv = np.indices((424,512))
grid_uv = grid_uv.reshape((2,-1))
grid_uv = np.vstack((grid_uv[1],grid_uv[0]))
grid = np.vstack((grid_uv,ones_array))

# Initialize texture map
texture_map = np.zeros((MAP['sizex'],MAP['sizey'],3)).astype(np.int16)

for count in range(np.size(d0)):
    
    # Convert depth image to metres and flatten the array (vectorization)
    depth_image = (d0[count]['depth']).reshape(1,-1)/1000

    # Remove noise (range of kinect v2 is about 4 to 5 metres)
    depth_image[depth_image>4.5] = 0
    
    # Get the time stamp for corresponding depth image
    depth_ts = d0[count]['t']
    
    # Get the coordinates in optical frame
    xy_o = np.matmul(IR_inv,grid)
    xyz_oir = np.multiply(xy_o,depth_image)
    # Homogenize the coordinates (add array of 1's)
    xyz_oir_homo = np.vstack((xyz_oir,ones_array))
    
    # Get lidar Timestamp  and hence cooresponding (x,y,theta) from particle filter
    index_lidar = (np.abs(lidar_TimeStamps-depth_ts)).argmin()
    current_x = current_positions[0][index_lidar]
    current_y = current_positions[1][index_lidar]
    current_theta = current_positions[2][index_lidar]
    
    # Get joint Timestamp and hence corresponding Joint Angles
    index = (np.abs(joint_TimeStamps-depth_ts)).argmin()

    # Get the transformation matrix for kinect to World Frame using current joint angles
    T_ktowf = get_cam_world_pose(joint_Angles[1][index],joint_Angles[0][index],current_x,current_y,current_theta)
    Transformation = np.matmul(T_ktowf,oRr_inv)
    # Get the xyz coordinates in World Frame
    xyz_wf = np.matmul(Transformation,xyz_oir_homo)

    # Get which coordinates belong to ground (Thresholding) 
    ground_truth = ((xyz_wf[2]>0) & (xyz_wf[2]<0.1))
    # Get the xyz coordinates for only ground
    xyz_ground = xyz_wf[:,ground_truth]
    # Reshape ground (for visualizing if algorithm is correct)
    ground = (ground_truth.astype(np.int8)).reshape(424,512)
    
    # Get the corresponding xyz coordinates in RGB frame using extrinics
    xyz_orgb = np.matmul(IR_RGB_inv,xyz_oir_homo[:,ground_truth])[:3,]
    # Get the corresponding pixels in RGB image
    uv_rgb = np.divide(np.matmul(RGB_intrinsics,xyz_orgb),xyz_orgb[2])[:2]
    # Force as integers
    uv_rgb_int = np.around(uv_rgb).astype(np.int16)
    # Find the feasible pixels, i.e. that can lie in RGB image (960x540)=(v,u)
    feasible = (uv_rgb_int[0]>=0) & (uv_rgb_int[1]>=0 ) & (uv_rgb_int[0]<960) & (uv_rgb_int[1]<540)
    # Get only feasible pixels
    uv_rgb_feasible = uv_rgb_int[:,feasible]
    # Get corresponding pixels in World Frame (Which belong to ground and have RGB color values)
    xyz_ground_feasible = xyz_ground[:,feasible]
    
    # Fill the texture map by calulating corresponding grid values and the filling corresponding colors
    xy_occ = np.ceil((xyz_ground_feasible[:2,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    texture_map[xy_occ[0],xy_occ[1],:] = r0[count]['image'][uv_rgb_feasible[1],uv_rgb_feasible[0],:]

# Read the environment's free space, occupied space and robot's trajectory generated from particle filter
occupancy_map = np.load('Occupancy4_texture.npy')
display_map = np.load('Display4_texture.npy')
pose_map = np.load('Poses4_texture.npy')

# Get the indices where map is free and where occupied
occ_x,occ_y = np.where(occupancy_map==1)
free_x,free_y = np.where(display_map==1)

# Merge the two maps i.e Free space map and Texture map for better visuals
texture_map_merged = np.zeros((MAP['sizex'],MAP['sizey'],3)).astype(np.int16) + 127
texture_map_merged[occ_x,occ_y,:] = 0
texture_map_merged[free_x,free_y,:] = 255
color_x,color_y,color_z = np.where((texture_map!=0) & (texture_map_merged==255) )
texture_map_merged[color_x,color_y,color_z] = texture_map[color_x,color_y,color_z]
texture_map_merged[pose_map[0].astype(np.int16),pose_map[1].astype(np.int16),0] = 255
texture_map_merged[pose_map[0].astype(np.int16),pose_map[1].astype(np.int16),1:3] = 0

plt.imshow(texture_map_merged)
plt.show()

