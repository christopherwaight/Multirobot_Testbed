%weight by distance to COM
function [W] = DCOM(NRobot,RobotParams,DL, SysParam)
N = floor(length(RobotParams)/4);
W = zeros(2,2*N); 

xs = RobotParams(1:4:end);
ys = RobotParams(2:4:end); 
SV = RobotParams(4:4:end); 
ds = sqrt((xs-xs(NRobot)).^2 + (ys-ys(NRobot)).^2); 
ds(ds>DL) =0; 

%to be careful specify all sensor values and positions of robots outside of sensor range
%as nan
xs(ds>DL) = NaN; 
ys(ds>DL) = NaN; 
SV(ds>DL) = NaN; 

%calculate local COM POS
n = sum(ds~=0); 
xcom = nansum(xs)/n; 
ycom = nansum(ys)/n; 

dcom = sqrt((xs-xcom).^2 + (ys-ycom).^2); 
W(1,1:2:end) = dcom.^2; 
W(2,2:2:end) = dcom.^2; 

end