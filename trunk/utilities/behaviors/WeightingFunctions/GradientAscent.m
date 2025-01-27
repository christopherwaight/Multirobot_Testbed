function [weights] = GradientAscent(NRobot,RobotParams,DL,~)
N = floor(length(RobotParams)/4);
SV = zeros(1,N);
SV(1,1:N) = RobotParams(4:4:end);
R = zeros(N,2); 
xs = RobotParams(1:4:end); 
ys = RobotParams(2:4:end);
R(:,1) = (xs-xs(NRobot))'; 
R(:,2) = (ys-ys(NRobot))';
d = sqrt(R(:,1).^2 + R(:,2).^2)';
R(d>DL,:) = 0; 
SV(d>DL) = 0; 
d(d>DL)=0; 
weights= zeros(2,2*N);
weights(1,1:2:2*N) = (SV-SV(NRobot)).*d;
weights(2,2:2:2*N) = (SV-SV(NRobot)).*d;
weights=(R'*R)\weights;


end