function [W] = ThetaGradient(NRobot,RobotParams,DL, SysParam)
N = floor(length(RobotParams)/4);
W = zeros(2,2*N); 
theta=0;  

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
dsmax = sqrt((xs-xs(SVmaxi)).^2 + (ys-ys(SVmaxi)).^2); %find distance between max robot and all other robots in sensor range
deltaSV = SVmax - SV;  % find delta SV btween max robot and all other robots
gx = dot(deltaSV, xs - xs(SVmaxi)); %estimate gradient
gy = dot(deltaSV, ys - ys(SVmaxi)); %estimate gradient
theta = atan2d(gy,gx); %find angle
ds(ds>DL)=0;

n = sum(ds~=0); 
%theta method 
alpha = 0.4; 
beta = 1-alpha; 
W(1,1:2:2*N) = (alpha*(sind(theta)^2 - cosd(theta)^2)+beta)*ds/n; 
W(1,2:2:2*N) = (alpha*-2 * sind(theta)*cosd(theta))*ds/n;
W(2,1:2:2*N) = (alpha*-2 * sind(theta)*cosd(theta))*ds/n; 
W(2,2:2:2*N) = (alpha*(cosd(theta)^2 - sind(theta)^2)+beta)*ds/n; 

end