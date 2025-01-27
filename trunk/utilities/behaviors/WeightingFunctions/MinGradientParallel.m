function [W] = MinGradientParallel(NRobot,RobotParams,DL,SysParam)
N = floor(length(RobotParams)/4);
W = zeros(2,2*N);  

xs = RobotParams(1:4:end);
ys = RobotParams(2:4:end); 
ds = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 
%ds(ds>DL) =0; 

%to be careful specify all sensor values and positions of robots outside of sensor range
%as nan
SV = RobotParams(4:4:end); 
SV(ds>DL) = NaN; 
xs(ds>DL) = NaN; 
ys(ds>DL) = NaN; 

[SVmax, SVmaxi] = max(SV); % find max robot
dsFromMax = sqrt((xs-xs(SVmaxi)).^2 + (ys-ys(SVmaxi)).^2); %find distance between max robot and all other robots in sensor range
delSFromMax = SVmax - SV; 
%Normalize ds and dx
maxdist = max(dsFromMax); 
maxdelS = max(delSFromMax); 


grad = (dsFromMax./maxdist)./(delSFromMax./maxdelS); %calculate gradient 
% find robot with minimum gradient from robot with max sensor value
[minGrad,minGradIdx] = min(grad);

W(1,SVmaxi*2-1) = -(ds(SVmaxi)); 
W(2,SVmaxi*2)   = -(ds(SVmaxi));
W(1,minGradIdx*2-1) = (ds(minGradIdx)); 
W(2,minGradIdx*2)   = (ds(minGradIdx));


end