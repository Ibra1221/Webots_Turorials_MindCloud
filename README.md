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
# Second The First tutorial📁 : 
* First we open webots from the terminal 

``webots``


this will open the program and then we make a new project by using the menu 

choose 

# Running turtlebot3 in webots as gazebo📹 :

``ros2 launch webots_ros2_turtlebot robot_launch.py``

In webots we use TwistStampted✅️ not Twist❌️

``ros2 run teleop_twist_keyboard teleop_twist_keyboard   --ros-args -p stamped:=true``
      
@ Copyrights MindCloud Robotics team,Faculty of Engineering ,Alexandria Universty,Egypt
