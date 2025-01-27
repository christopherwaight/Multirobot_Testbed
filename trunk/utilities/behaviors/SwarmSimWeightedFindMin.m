function [Vf] = SwarmSimWeightedFindMin(RobotParams, NRobot, SensorRange)
%SwarmSimFindMin
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
% %V_attract_r = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@OnesWeights,ref);
% V_attract = -SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@SVminusRef,ref);
% %V_disperse = SwarmSimWeightedDisperseNew(RobotParams, NRobot, SensorRange,@HigherSVThanRefWeights,ref);
% %VfV = 0.5*V_attract + 0.25*V_disperse + 0.25*V_attract_r; 
% VfV = V_attract;

V_attract = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@OnesWeights,[]);
V_attract_r = -SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@HigherSVThanRefWeights,ref);
%V_disperse = SwarmSimWeightedDisperseNew(RobotParams, NRobot, SensorRange,@LowerSVThanRefWeights,ref);

VfV = 0.5*V_attract + 0.5*V_attract_r;
n = norm(VfV);
if n ~=0 
    Vf = VfV/n; 
end

end
