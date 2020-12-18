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

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


# Function for Stratified Resampling
def Resmapling_particles(robot_currentPoseX_particles,robot_currentPoseY_particles,robot_currentPoseTheta_particles,particle_weights):

    j = 0 
    c = particle_weights[0]
    N = np.size(particle_weights)
    resampled_particlesX = np.zeros((N,1))
    resampled_particlesY = np.zeros((N,1))
    resampled_particlesTheta = np.zeros((N,1))
    
    for k in range(N):

        u = np.random.uniform(0,1/N)
        beta = u + k/N
        
        while beta > c:
            j = j+1
            c = c + particle_weights[j]
            # print(f'beta = {beta} & c = {c}')
        resampled_particlesX[k] = robot_currentPoseX_particles[j]
        resampled_particlesY[k] = robot_currentPoseY_particles[j]
        resampled_particlesTheta[k] = robot_currentPoseTheta_particles[j]

    particle_weights = np.zeros((N,1)) + (1/N)
    
    return resampled_particlesX, resampled_particlesY, resampled_particlesTheta, particle_weights





## SymPy Implementation for transformation
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
    [0,0,1,0.48],
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

T_laser = Matrix([[lx],
                  [ly],
                  [0],
                  [1]])
T_final = T_full*T_laser
point_in_WF = sympy.simplify(T_final)

# Lambda function for faster computations
point_fast = lambdify((head_angle,neck_angle, x, y, pose_angle, lx, ly),point_in_WF)

# init MAP
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -20  #meters
MAP['ymin']  = -20
MAP['xmax']  =  20
MAP['ymax']  =  20 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
MAP['logodds'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int16)
MAP['img'] = np.zeros((MAP['sizex'],MAP['sizey'],3),dtype=np.int8) #DATA TYPE: char or int8
MAP['display'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) 

# Initializing Map correlation properties
x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
x_max_range = 0.2
y_max_range = 0.2
x_mid = int(x_max_range/MAP['res'])
y_mid = int(y_max_range/MAP['res'])
x_range = np.arange(-x_max_range,x_max_range+MAP['res'],MAP['res'])
y_range = np.arange(-y_max_range,y_max_range+MAP['res'],MAP['res'])

# Set initial robot position to (0,0,0)
robot_currentPoseX = 0
robot_currentPoseY = 0
robot_currentPoseTheta = 0

# Initialize Particle filter parameters
N_particles = 200
N_threshold = 100
robot_currentPoseX_particles = np.zeros((N_particles,1))
robot_currentPoseY_particles = np.zeros((N_particles,1))
robot_currentPoseTheta_particles = np.zeros((N_particles,1))
particle_weights = np.zeros((N_particles,1)) + (1/N_particles)
x_max_correlation = np.zeros((N_particles,1))
y_max_correlation = np.zeros((N_particles,1))
correlation_value = np.zeros((N_particles,1))

# Overfitting parameters
upper_threshold = 10
lower_threshold = -10

# Load data
j0 = data.get_joint("joint/train_joint0")
l0 = data.get_lidar("lidar/train_lidar0")
joint_TimeStamps = j0['ts']
joint_Angles = j0['head_angles']

# Initialize parameters required for cv2.line()
color = (255,255,255)
thickness = 1 #px

# For storing robot positions in grids and in metres
MAP['pose'] = np.zeros((2,np.size(l0)))
positions = np.zeros((3,np.size(l0)))

for count in range(np.size(l0)):

    print(count)
    d = l0[count]
    lidar_TimeStamp = d['t']
    lidar_deltaPose = d['delta_pose']
    lidar_Data = d['scan']

    # Time synchronization
    index = (np.abs(joint_TimeStamps-lidar_TimeStamp)).argmin()

    # Get valid LIDAR data
    theta_lidar = np.arange(-135,135.25,0.25)*np.pi/180
    indValid = np.logical_and(( lidar_Data[0]< 30),(lidar_Data[0]> 0.1))
    lidar_Data = (lidar_Data[0][indValid]).reshape((1,np.sum(indValid)))
    theta_lidar = theta_lidar[indValid]

    # (x,y) coordinates in LIDAR frame
    xs0 = lidar_Data*np.cos(theta_lidar)
    ys0 = lidar_Data*np.sin(theta_lidar)

    # Initialization for the very 1st time step
    if count==0:

        # Get the grid values of current position
        robot_currentGridX = (np.ceil((robot_currentPoseX - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1).astype(np.int16)
        robot_currentGridY = (np.ceil((robot_currentPoseY - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1).astype(np.int16)

        # Get the LIDAR data in World Frame
        calculated_points = point_fast(joint_Angles[0][index],joint_Angles[1][index],robot_currentPoseX,robot_currentPoseY,robot_currentPoseTheta,xs0[0],ys0[0])
        x_WF = calculated_points[0][0]
        y_WF = calculated_points[1][0]
        z_WF = calculated_points[2][0]

        # Convert LIDAR scans to grid
        xis = np.ceil((x_WF - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((y_WF - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    # After 1st time step
    else:

        # Using Particle Filter

        # Motion model + noise
        robot_currentPoseX_particles = robot_currentPoseX_particles + lidar_deltaPose[0][0] + np.random.normal(0,0.05,(N_particles,1))
        robot_currentPoseY_particles = robot_currentPoseY_particles + lidar_deltaPose[0][1] + np.random.normal(0,0.05,(N_particles,1))
        robot_currentPoseTheta_particles = robot_currentPoseTheta_particles + lidar_deltaPose[0][2] + np.random.normal(0,0.025,(N_particles,1))

        # Get the LIDAR data in World Frame
        calculated_points = point_fast(joint_Angles[1][index],joint_Angles[0][index],robot_currentPoseX_particles,robot_currentPoseY_particles,robot_currentPoseTheta_particles,xs0[0],ys0[0])
        x_WF = calculated_points[0][0]
        y_WF = calculated_points[1][0]
        z_WF = calculated_points[2][0]
             
        # For each particle, get MAP correlation
        for part_n in range(N_particles):

            Y = np.vstack([x_WF[part_n],y_WF[part_n]])
            c_ex = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)

            # Get the index with maximum map correlation
            ind = np.unravel_index(np.argmax(c_ex, axis=None), c_ex.shape)

            # Store the maximum correlation accros grid for this particle
            correlation_value[part_n] = c_ex[ind]

            # Update the particle's position according to maximum correlation value
            x_max_correlation[part_n] = (ind[0]-x_mid)*MAP['res'] + robot_currentPoseX_particles[part_n]
            y_max_correlation[part_n] = (ind[0]-y_mid)*MAP['res'] + robot_currentPoseY_particles[part_n]


        # Choose best particle
        ind = np.unravel_index(np.argmax(correlation_value, axis=None), correlation_value.shape)

        # Update robot's position according to best particle's position
        robot_currentPoseX = x_max_correlation[ind[0]][0]
        robot_currentPoseY = y_max_correlation[ind[0]][0]

        # Convert the LIDAR scans as seen by the best particle to grid
        xis = np.ceil((x_WF[ind[0]] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((y_WF[ind[0]] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1


        # To avoid noise and map size issues consider only those indices which are within the MAP size
        isGoodX = np.logical_and((xis < MAP['sizex']),(xis >= 0))
        isGoodY = np.logical_and((yis < MAP['sizey']),(yis >= 0))

        # Avoid ground plane 
        isGoodZ = (z_WF > 0.1)
        isGooD1 = np.logical_and(isGoodX,isGoodY)
        isGooD = np.logical_and(isGooD1,isGoodZ)
        xis = xis[isGooD]
        yis = yis[isGooD]
        z_WF = z_WF[isGooD]
        
        # Update particle weight using softmax function
        particle_weights = np.multiply(particle_weights, np.exp(correlation_value-np.amax(correlation_value)))
        # Normalize particles' weight
        particle_weights = particle_weights/np.sum(particle_weights)

        # Check if resampling is needed using effective number of particles
        # Since we have already stored best particle's position LIDAR scans etc, we need not worry about location
            # of Resampling in the code now 
        N_eff = 1/np.sum(np.square(particle_weights))
        if N_eff <= N_threshold:
            robot_currentPoseX_particles,robot_currentPoseY_particles,robot_currentPoseTheta_particles,particle_weights = Resmapling_particles(robot_currentPoseX_particles,robot_currentPoseY_particles,robot_currentPoseTheta_particles,particle_weights)

        # Calculate robot's positions in grid cells
        robot_currentGridX = (np.ceil((robot_currentPoseX - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1).astype(np.int16)
        robot_currentGridY = (np.ceil((robot_currentPoseY - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1).astype(np.int16)
        
        # Store positions (Just for offline viewing, later used for texture mapping)
            # Store (x,y,theta) positions in metres and radians 
        positions[0][count] = robot_currentPoseX
        positions[1][count] = robot_currentPoseY
        positions[2][count] = robot_currentPoseTheta_particles[ind[0]][0]
            # Store (x,y) positions in terms of grid cells
        MAP['pose'][0][count] = robot_currentGridX
        MAP['pose'][1][count] = robot_currentGridY

    # Get free cells using cv2.line()
    for i in range(np.size(xis)):
        cv2.line(MAP['img'],(robot_currentGridY,robot_currentGridX), (yis[i],xis[i]), color)

    # Maintain a log odds map
    MAP['logodds'] -= ((MAP['img'][:,:,0]/127)*(np.log(4))).astype(np.int16)
    MAP['logodds'][xis,yis] += np.log(4) 
    # Fill 0 so it can be reused for next LIDAR scan, i.e. next timestamp
    MAP['img'].fill(0)
    # Get binary occupancy map (wherever log odds > 0)
    MAP['map'] = (MAP['logodds'] > 0).astype(np.int8)
    
    # Avoiding overfitting
    high_idices = MAP['logodds'] > upper_threshold
    MAP['logodds'][high_idices] = upper_threshold
    low_indices = MAP['logodds'] < lower_threshold
    MAP['logodds'][low_indices] = lower_threshold

# Get the free space map (wherever log odds < 0)
MAP['display'] = (MAP['logodds'] < 0).astype(np.int8)

# Save files for offline use (viewing, texture mapping etc)
np.save('Display0_texture.npy',MAP['display'])
np.save('Poses0_texture.npy',MAP['pose'])
np.save('Occupancy0_texture.npy',MAP['map'])
np.save('Positions0_texture.npy',positions)

# Plot the final result
fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(121)
plt.imshow(MAP['map'],cmap="hot")
plt.title("Occupancy map")

ax2 = fig.add_subplot(122)
plt.scatter(MAP['pose'][1],MAP['pose'][0],s = 1,c='r')
plt.imshow(MAP['display'],cmap="hot")
plt.title("Free space map")


plt.show()
