function [W] = EllipseMinMaxDisperse(NRobot,RobotParams,DL,SysParam)
N = floor(length(RobotParams)/4);
SV = zeros(2,2*N); 
%Only need 2 specified parameters for ellipse, semi major axis and
%eccentricity
a = SysParam(1); 
e = SysParam(2);
% b is calculated using the eccentricity and semimajor axis
b = a * sqrt(1-e^2);


xs = RobotParams(1:4:end);
ys = RobotParams(2:4:end); 
ss = RobotParams(4:4:end);
ds = zeros(N,1); 
ds = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 
xs(ds>DL) = nan; 
ys(ds>DL) = nan; 
ds(ds>DL) = nan; 
ss(ds>DL) = nan;

%alpha is determined as angle from max robot to min robot
[~, maxSi] = max(ss); 
[~, minSi] = min(ss); 
alpha= 0; 
alpha = atan2d(ys(minSi) - ys(maxSi), xs(minSi) - xs(maxSi)); 

%xc and yc are calculated using a, alpha, and the coordinates of max robot
xc = 0; 
yc = 0; 
%xc = xs(maxSi) + a*cosd(alpha); 
%yc = ys(maxSi) + a*sind(alpha); 
xc = mean(xs); 
yc = mean(ys);

%now we have all five parameters to see if NRobot is in the ellipse
EQ = (((xs(NRobot)-xc)*cosd(alpha)+ (ys(NRobot)-yc)*sind(alpha)).^2 )./ (a^2) + (((xs(NRobot)-xc)*sind(alpha)- (ys(NRobot)-yc)*cosd(alpha)).^2 )./ (b^2);

W = zeros(2,2*N); 
if EQ <1 
%     [~,mini] = mink(ds,2);  
%     w(mini) = -ds(mini);
else
    W(1, 1:2:end) = (a*sind(alpha)^2/b+b*cosd(alpha)^2/a)*ds;
    W(1, 2:2:end) = cosd(alpha)*sind(alpha)*(-a/b+b/a)*ds;
    W(2, 1:2:end) = cosd(alpha)*sind(alpha)*(-a/b+b/a)*ds;
    W(2, 2:2:end) = (a*cosd(alpha)^2/b+b*sind(alpha)^2/a)*ds;
end

W(isnan(W)) = 0;

end