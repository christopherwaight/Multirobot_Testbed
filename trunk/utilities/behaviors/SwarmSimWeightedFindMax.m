function [Vf] = SwarmSimWeightedFindMax(RobotParams, NRobot, SensorRange)
%SwarmSimFindMax Find Max behavior called in Find Max block of Swarm_Robot_Base
%        MOST RECENT UPDATE: 09/18/2018 by NJM
%   SwarmSimFindMax takes the inputs of RobotParams, NRobot, and Sensor
%   Range and outputs the resultant velocity. Individual velocity of robot
%   is determined by comparing "Sensor Value" to surrounding robots and
%   moving in the direction of higher readings.

%% Initialize Variables
% Determine number of robots based off length of robot params vector 
N= floor(length(RobotParams)/4); 
 
% position variables for 3-DOF omnibot are x,y,and theta (rotation about z-axis)
S=zeros(1,N);
% Vf - final velocity as a vector.
Vf=[0.0 0.0 0.0];


%% Set x and y inputs into an array
S(1,1:N)=RobotParams(4:4:end);
ref = S(NRobot); 
%Vf = SwarmSimWeightedAttractNew(RobotParams,NRobot,SensorRange,@GradientAscent,SensorRange);
V_attract = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@EllipseDisperseTangent,[10,0,0]);
V_disp = SwarmSimWeightedAttractNew(RobotParams,NRobot,SensorRange,@distBtwnRobots,[]); 
V_attract_r = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@HigherSVThanRefWeights,ref);
%V_disperse = SwarmSimWeightedDisperseNew(RobotParams, NRobot, SensorRange,@LowerSVThanRefWeights,ref);

VfV = 0.5*V_attract -0.3*V_disp + 0.2*V_attract_r;
%VfV = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@SimpleGradientCalc,ref);
%VfV = 0.5*V_attract + 0.5*V_attract_r; 
n = norm(VfV);
if n ~=0 
    Vf = VfV/n; 
end

end
