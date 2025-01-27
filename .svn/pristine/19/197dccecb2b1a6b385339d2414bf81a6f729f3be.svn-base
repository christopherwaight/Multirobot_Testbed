function [W] = comAttract(NRobot,RobotParams,DL,~)
N = floor(length(RobotParams)/4);
W = zeros(2,2*N); 
xs = zeros(1,N); 
ys = zeros(1,N);
ds = zeros(1,N); 
xs(1:N) = RobotParams(1:4:end); 
ys(1:N) = RobotParams(2:4:end); 
ds(1:N) = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 

thetad = 90;
T = [cosd(2*thetad), -sind(2*thetad); sind(2*thetad) cosd(2*thetad)]; 
T = [cosd(thetad), -sind(thetad); sind(thetad) cosd(thetad)]; 
Ws = T*[0.1 0 ; 0 0.9]; 
W(1,1:2:2*N) = ds/N *Ws(1,1);
W(1,2:2:2*N) = ds/N *Ws(1,2);
W(1,2:2:2*N) = ds/N *Ws(2,1);
W(2,2:2:2*N) = ds/N *Ws(2,2);


% W(1,1:2:2*N) = ds/N ;
% W(2,1:2:2*N) = -ds/N*10;
% W(2,2:2:2*N) = ds/N;