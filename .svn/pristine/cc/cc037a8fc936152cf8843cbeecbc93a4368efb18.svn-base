function [weights] = CenteronSV(NRobot,RobotParams,DL, SimParams)
N = floor(length(RobotParams)/4);
xs = RobotParams(1:4:end); 
ys = RobotParams(2:4:end); 
ss = RobotParams(4:4:end); 
% remove robots outside of sensor range
ds = sqrt((xs - xs(NRobot)).^2 + (ys- ys(NRobot)).^2); 
xs(ds>DL) = nan; 
ys(ds>DL) = nan; 
ss(ds>DL) = nan; 


%Find min, max, and medain sensor value value
[sMin, ~] = min(ss); 
[sMax, ~] = max(ss); 
[sMed] = median(ss); 

w = ((sMin+sMax)/2 - sMed).*(ss - ss(NRobot)); 
w(isnan(w)) =0;


weights= zeros(2,2*N);
weights(1,1:2:2*N) = w;
weights(2,2:2:2*N) = w;

end