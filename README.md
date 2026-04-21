# Webots_Turorials_MindCloud🔥
This repo contains projects and guide for beginners in webots simulation program &amp; include troubleshooting for most common errors in webots
# First to install ⬇️: 
* Using to Approaches :
  1. Using snap :
     ``sudo snap install webots --classic``
  3. Using Dep :
     Download from the website : https://cyberbotics.com/#download
     
     choose the Ubuntu Debain (.deb)
     
     ``cd ~/Downloads``
     
     ``sudo dpkg -i webots_2025a_amd64.deb``
     
     ``sudo apt -f install``
     
     ``webots``
     
* TroubleShooting for Uninstalling :
  
    ``sudo snap remove webots``

# Explaining the repo Content⚡️ :
 First the webots_tutorial_1 :
 
 contains :
 
 1- vo code **node** 
 
 2- contains GPS+IMU+Camera Depth&RGB **Sensors** 
 
 3- Costmap Code **node**
 
 4- Controllers **node**
 
 5- urdf files **urdf**
 
 6- wbt **webots_worlds**
 
 7- **launch files**
 
# Second The First tutorial📁(Running the simulation for the Nexsuis Rover (New Rover)) :

  Open your Terminal :
  
  ``cd webots_tutotrial_1/``

  ``source install/setup.bash``
  
  ``ros2 launch webots_pkg_sim sim_rover_launch.py``

  This will open the Simulation

  Now you have the Simulation opened in **Webots**

  Let's run the rover !

  in another Terminal :

  ``ros2 run teleop_twist_keyboard teleop_twist_keyboard   --ros-args -p stamped:=true``

  Use the arrows to move it.

  Now let's Jump to the second terminal 

  # Second tutorial (Using the Costmap Code And Testing it):

  Open your Terminal :

  After running the previous tutorial 

  source the workspace in another Terminal 
  
  ``ros2 run webots_pkg_sim costmap_lifecycle``

  then start to configure and activate the lifecycle node 

  Open Rviz2 :

  ``source ~/webots_tutorial_1/install/setup.bash``
  
  ``rviz2``

  then choose the Fixed Frame >> base_link

  then choose the Map in the rviz and choose the theme **costmap**

  then choose the **PointCloud2** 

  try to test the code by moving the rover and move the boxes to test it

  Here is a video demo for all this :

<video src="" controls="controls" style="max-width: 100%;">
</video>

  



this will open the program and then we make a new project by using the menu 

choose 

# Running turtlebot3 in webots as gazebo📹 :

``ros2 launch webots_ros2_turtlebot robot_launch.py``

In webots we use TwistStampted✅️ not Twist❌️

``ros2 run teleop_twist_keyboard teleop_twist_keyboard   --ros-args -p stamped:=true``
      
@ Copyrights MindCloud Robotics team,Faculty of Engineering ,Alexandria Universty,Egypt
