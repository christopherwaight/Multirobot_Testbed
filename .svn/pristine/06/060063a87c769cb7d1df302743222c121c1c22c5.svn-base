function [contourState] = SwarmSimOnContour(RobotParams, NRobot, DesiredValue)
% SWARMSIMONCOUNTER - <Determines if a robot is above, below, or on a
% desired contour value.>

% Outputs:
%   contourState     1=below contour, 2= above contour, 3= on contour

%% Initialize Variables

% Determine number of robots based off length of robot params vector 
N= floor(length(RobotParams)/4); 

%% initialize variables 
% position variables for 3-DOF omnibot are x,y,and theta (rotation about z-axis)
SensorValue= zeros(1,N); % this is the value of the "sensor reading" from each robot's "on-board" sensor. 

%% Set SensorValue inputs into an array
SensorValue(1,1:N)=RobotParams(4:4:4*N);

%% Algorithm

% Grab robot's sensor reading
robotValue=SensorValue(NRobot); 

% Check to see if robot is close to desired contour, below, or above
% contour
if robotValue == DesiredValue
    %On Contour
    contourState=3;
    return;
elseif robotValue < DesiredValue
    %Below Contour
    contourState=1;
    return;
else  %robotValue > DesiredValue
    %Above Contour
    contourState=2;
    return;
end


end