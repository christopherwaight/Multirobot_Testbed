function [W] = donutAttract(NRobot,RobotParams,DL,SysParams)
N = floor(length(RobotParams)/4);
W = zeros(2,2*N); 
xs = zeros(1,N); 
ys = zeros(1,N);
ds = zeros(1,N); 
%remove robots too far away
xs(1:N) = RobotParams(1:4:end); 
ys(1:N) = RobotParams(2:4:end); 
ds(1:N) = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 
xs(ds>DL) = nan; 
ys(ds>DL) = nan; 
ds(ds>DL) = nan; 
n_inrange = sum(~isnan(ds)); 

% calculate COM Position
xcom = nansum(xs)/n_inrange; 
ycom = nansum(ys)/n_inrange; 
d_com = sqrt( (xcom-xs(NRobot))^2 + (ycom-ys(NRobot))^2); 
%Define inner and outer donut ranges
a = 15; 
b = 25; 
% invds = 1./ds; 
% invds(NRobot) = 0; 
% W(1,1:2:2*N) = floor((d_com-a)/(b-a))-invds;
% W(2,2:2:2*N) = floor((d_com-a)/(b-a))-invds; 
W(1,1:2:2*N) = floor((d_com-a)/(b-a));
W(2,2:2:2*N) = floor((d_com-a)/(b-a));

end