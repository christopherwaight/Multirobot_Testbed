function [weights] = SVminusRef(NRobot,RobotParams,DL, Reference)  
N = floor(length(RobotParams)/4);
SV = zeros(1,N);
SV(1,1:N) = RobotParams(4:4:end);
weights= zeros(2,2*N);
weights(1,1:2:2*N) = SV-Reference;
weights(2,2:2:2*N) = SV-Reference;

end