function [Vf] = SwarmSimObstacleAvoid(RobotParams, NRobot, SensorRange, AvoidRange)
%SwarmSimObstacleAvoid Obstacle Avoidance behavior for Swarm_Adaptive_Navigation_Simulator.slx
%   LATEST UPDATE: 09/06/2018 by NJM 
% This obstacle avoidance behavior is called by the Robot # behavior blocks in
%   Swarm_Adaptive_Naviagtion_Simulator/ObstacleAvoidance blocks. The
%   obstacle avoidance behavior causes robots to move away from other 
%   robots (or objects) that are sensed to be in the robot's avoidance
%   radius, which is an input for this function. The behavior is designed
%   to be non-linear, so that avoidance is damped outside of the
%   AvoidRange, but is the "strongest" component when the robot is close to
%   other robots or objects. 

% Determine number of robots based off length of robot params vector 
N= floor(length(RobotParams)/4);

%% Initialize Variables 
% position variables for 3-DOF omnibot are x and y
x=zeros(1,N);               
y=zeros(1,N);
% d - distance between "self" and other robot
% O - bearing angle between "self" and other robot
d=zeros(1,N);
O=zeros(1,N);
mag_velocity=zeros(1,N);
Vft=0;
% Vf - final velocity as a vector.
Vf=[0.0 0.0 0.0];

%% Set x,y,theta, and SensorValue inputs into an array
x(1,1:N)=RobotParams(1:4:4*N);
y(1,1:N)=RobotParams(2:4:4*N);

%% Obstacle Avoidance Behavior 

% Determine distance and angle to each robot 
d(1,1:N) = sqrt( ( x(NRobot)-x ).^2 + ( y(NRobot)-y ).^2 ); 
O(1,1:N) = atan2((y-y(NRobot)),(x-x(NRobot)));

% Calculate avoidance velocities 
idx_in_ObsRange = find(d<=AvoidRange & d~=0); 
mag_velocity(idx_in_ObsRange) = 100*(AvoidRange./d(idx_in_ObsRange)).^4;
mag_velocity(mag_velocity>0.5*realmax) = 0.5*realmax;
Vfx = sum(mag_velocity.*cos(O)); 
Vfy = sum(mag_velocity.*sin(O)); 

Vf(1,1:3)= -[Vfx, Vfy, Vft]; 
end 
 

    

