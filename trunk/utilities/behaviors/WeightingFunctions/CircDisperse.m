function [W] = CircDisperse(NRobot,RobotParams,DL,SysParam)
N = floor(length(RobotParams)/4);
SV = zeros(2,2*N); 
R = 10; %m

xs = RobotParams(1:4:end);
ys = RobotParams(2:4:end); 
ss = RobotParams(4:4:end);
ds = zeros(N,1); 
ds = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 
ds(ds>DL) =nan; 

W = zeros(2,2*N); 
w1 = zeros(1,N); 
w2 = zeros(1,N); 
w1(ds>R) = ds(ds>R); 
[~,mini] = mink(ds,2);  
w2(mini) = -ds(mini);
w = w1+w2; 
w(isnan(w)) = 0;

W(1,1:2:end) = w1+w2; 
W(2,2:2:end) = w1+w2; 
end