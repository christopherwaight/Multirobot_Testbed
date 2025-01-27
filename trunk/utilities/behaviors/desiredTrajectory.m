function [xd, yd] = desiredTrajectory(time)
%         xd = 0.005*Robot_Data.out.tout;
%         yd = 8*sin(0.005*Robot_Data.out.tout);
%         xd = 0.005*time;
%         yd = 0.75*sin(0.1*time);

% % xd = 0.005*t;
% % yd = 0.75*sin(0.1*t);
a = 0.05;
M = 1;
b = 0.1;

if time < 45
    xd = 0.5;
else
    xd = 0.75*(time -45)/60 + 0.5; 
end
yd = 0; 
% xd = a*time;
% yd = M*sin(b*time); 
% xd = 0.1*time;
% yd = nan; 
end