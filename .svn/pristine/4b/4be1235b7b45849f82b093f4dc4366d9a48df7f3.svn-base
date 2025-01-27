function [W] = distRelativeToD(NRobot,RobotParams,DL,SymParams)
N = floor(length(RobotParams)/4);
W = zeros(2,2*N); 
xs = zeros(1,N); 
ys = zeros(1,N);
ds = zeros(1,N); 
d = 2; %m

xs(1:N) = RobotParams(1:4:end); 
ys(1:N) = RobotParams(2:4:end); 
ds(1:N) = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 

w = ds; 
w(ds>DL) = 0; 
    
W(1,1:2:2*N) = w;
W(2,2:2:2*N) = w; 


end
