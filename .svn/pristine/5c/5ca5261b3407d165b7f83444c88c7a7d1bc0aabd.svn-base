function [W] = EllipseDisperseTangent(NRobot,RobotParams,DL,SysParam)
N = floor(length(RobotParams)/4);
SV = zeros(2,2*N); 
% Need three parameters specified: semimajor axis (a), eccentricity (e), 
% and angle of ellipse (alpha)
% a = 25; %m 
% e =0.75; 
a = SysParam(1); 
e = SysParam(2);
alpha = SysParam(3)+45; 

% semiminor axis(b) is specified by eccentricity
b = a * sqrt(1-e^2);

ds = zeros(N,1); 
xs = zeros(N,1); 
ys = zeros(N,1); 
ss = zeros(N,1); 

xs = RobotParams(1:4:end);
ys = RobotParams(2:4:end); 
ss = RobotParams(4:4:end);

ds = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 
xs(ds>DL) = nan; 
ys(ds>DL) = nan; 
ds(ds>DL) = nan; 

%center of ellipse is specified by the COM (xc,yc)
xc = mean(xs,'omitnan');
yc = mean(ys,'omitnan'); 

EQ = (((xs(NRobot)-xc)*cosd(alpha)+ (ys(NRobot)-yc)*sind(alpha)).^2 )./ (a^2) + (((xs(NRobot)-xc)*sind(alpha)- (ys(NRobot)-yc)*cosd(alpha)).^2 )./ (b^2);

W = zeros(2,2*N); 
if EQ <=1
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