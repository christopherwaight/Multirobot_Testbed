function [W] = ThetaMinG(NRobot,RobotParams,DL,SysParam)
N = floor(length(RobotParams)/4);
W = zeros(2,2*N); 

xs = RobotParams(1:4:end);
ys = RobotParams(2:4:end); 
ds = zeros(N,1); 
ds = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 
ds(ds>DL) =0; 
n = sum(ds~=0);

 
%to be careful specify all sensor values and positions of robots outside of sensor range
%as nan
SV = RobotParams(4:4:end); 
SV(ds>DL) = NaN; 
xs(ds>DL) = NaN; 
ys(ds>DL) = NaN; 

[SVmax, SVmaxi] = max(SV); % find max robot
dsFromMax = sqrt((xs-xs(SVmaxi)).^2 + (ys-ys(SVmaxi)).^2); %find distance between max robot and all other robots in sensor range
delSFromMax = SVmax- SV; 
[minG, minGi] = min(delSFromMax./dsFromMax); 


%theta method 
theta=0;
theta=atan2d(ys(SVmaxi)- ys(minGi), xs(SVmaxi) - xs(minGi))+90; 
if length(theta)>1
    theta= theta(1);
end
%theta =  (-180/10000)*SysParam(2)+90; 
alpha = .45; 
beta =1-alpha; 
W(1,1:2:2*N) = (alpha*(sind(theta)^2 - cosd(theta)^2)+beta)*ds/n; 
W(1,2:2:2*N) = (alpha*-2 * sind(theta)*cosd(theta))*ds/n;
W(2,1:2:2*N) = (alpha*-2 * sind(theta)*cosd(theta))*ds/n; 
W(2,2:2:2*N) = (alpha*(cosd(theta)^2 - sind(theta)^2)+beta)*ds/n; 

end