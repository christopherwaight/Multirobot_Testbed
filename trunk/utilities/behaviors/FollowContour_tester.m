function Vf  = FollowContour_tester(RobotParams, NRobot, SensorRange, contourState,DesiredValue)
% calls the correct behavior to follow contour
Vf = [ 0.0 0.0 0.0]; 
VCW = [ 0 0 0]; 
VCCW = [0 0 0]; 
V_attract = [0 0 0]; 
switch contourState
    case 1
        %Below the contour
        VCW = SwarmSimWeightedCWOrbitNew(RobotParams, NRobot, SensorRange,@HigherSVThanRefWeights,DesiredValue);
        V_attract_r = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@OnesWeights,DesiredValue);
        V_attract = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@HigherSVThanRefWeights,DesiredValue);
        V_disperse = SwarmSimWeightedDisperseNew(RobotParams, NRobot, SensorRange,@LowerSVThanRefWeights,DesiredValue);
        %VfV = 0.3*VCW+ 0.3*V_attract + 0.1*V_disperse+ 0.3*V_attract_r; 
        VfV = 0.3*VCW+ 0.4*V_attract + 0.3*V_attract_r; 
        
    case 2
        %Above the contour
        VCCW = SwarmSimWeightedCCWOrbitNew(RobotParams, NRobot, SensorRange,@LowerSVThanRefWeights,DesiredValue);
        V_attract_r = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@OnesWeights,DesiredValue);
        V_attract = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@LowerSVThanRefWeights,DesiredValue);
        V_disperse = SwarmSimWeightedDisperseNew(RobotParams, NRobot, SensorRange,@HigherSVThanRefWeights,DesiredValue);
        %VfV = 0.3*VCW+ 0.3*V_attract + 0.1*V_disperse+ 0.3*V_attract_r; 
        VfV = 0.3*VCW+ 0.4*V_attract + 0.3*V_attract_r; 
    otherwise
        V_attract_r = SwarmSimWeightedAttractNew(RobotParams, NRobot, SensorRange,@OnesWeights,DesiredValue);
        VCW = SwarmSimWeightedCWOrbitNew(RobotParams, NRobot, SensorRange,@HigherSVThanRefWeights,DesiredValue);
        VCCW = SwarmSimWeightedCCWOrbitNew(RobotParams, NRobot, SensorRange,@LowerSVThanRefWeights,DesiredValue);
        VfV = 0.4*VCW+ 0.4*VCCW +0.2*V_attract_r;
end
n = norm(VfV);
if n ~=0 
    Vf = VfV/n; 
end
end
