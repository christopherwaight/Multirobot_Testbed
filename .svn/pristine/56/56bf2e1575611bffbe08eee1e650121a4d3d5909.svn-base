function [Vf] = SwarmSimFormation(RobotParams, NRobot, SensorRange,Distances)
%SwarmSimFormation
%% Initialize Variables
% Determine number of robots based off length of robot params vector 
N= floor(length(RobotParams)/4); 
 
% position variables for 3-DOF omnibot are x,y,and theta (rotation about z-axis)
% Vf - final velocity as a vector.
Vf=[0.0 0.0 0.0];

%% Set x and y inputs into an array
[VfV] = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@distRelativeToSetpoint, Distances);
n = norm(VfV);
if n ~=0 
    Vf = VfV/n; 
end

end