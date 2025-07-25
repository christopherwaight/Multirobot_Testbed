function [Vf] = SwarmSimWeightedCCWOrbitNew(RobotParams, NRobot, SensorRange,WeightFunction,SysParam)
%SwarmSimWeightedCCWOrbitNew Weighted CCW Orbit behavior for Swarm_Adaptive_Navigation_Simulator.slx
%   LATEST UPDATE: 08/30/2018 by NJM 
%   This attract behavior is called used by other composite behaviors in 
%   Swarm_Adaptive_Naviagtion_Simulator. The weighted attract behavior
%   causes the robots to clump together but favors weighting toward certain
%   robots. It is set for constant velocity and for constant force. 

%   WeightFunctions take a robot number for "self" and the robot params and
%   need to return a 1 by N vector of weights


% Determine number of robots based off length of robot params vector 
N= floor(length(RobotParams)/4); 

%% initialize variables 
% position variables for 3-DOF omnibot are x,y,and theta (rotation about z-axis)
x = zeros(2*N,1); 

% d - distance between "self" and other robot
d=zeros(1,2*N); 

%Vft is final angular velocity about z
Vft=0;
% Vf - final velocity as a vector.
Vf=[0.0 0.0 0.0];


%% Set x and y inputs into an array
x(1:2:end) = RobotParams(1:4:end); 
x(2:2:end) = RobotParams(2:4:end); 
d(1:2:2*N) = sqrt((x(1:2:end)-x(2*NRobot-1)).^2 + (x(2:2:end)-x(2*NRobot)).^2); 
d(2:2:2*N) = sqrt((x(1:2:end)-x(2*NRobot-1)).^2 + (x(2:2:end)-x(2*NRobot)).^2); 
d(d>SensorRange| d==0)=inf;
P = diag(1./d);

%% Attract Behavior 

% Determine distance and angle to each robot 
L = eye(2*N);
L(1:2:end,2*NRobot-1) = L(1:2:2*N,2*NRobot-1)-1;
L(2:2:end,2*NRobot) = L(2:2:2*N,2*NRobot)-1;
W = WeightFunction(NRobot,RobotParams,SysParam);

% If robot is within sensor range of  Nrobot, it is attracted to that
% robot. If the  robot is outside of sensor range, then that robot has
% no effect on NRobot's velocity. 
Kx = [ 0 1; -1 0]; 
v = Kx*W*P*L*x; 
%Calculate velocity vector components: 
Vfx= v(1);
Vfy= v(2); 
mag = sqrt(Vfx^2+Vfy^2); 

Vf(1,1:3)= [Vfx Vfy Vft]/mag;
if isnan(sum(Vf))
    Vf = [0 0 0];
end


end 



