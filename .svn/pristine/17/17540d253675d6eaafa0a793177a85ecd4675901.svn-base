function [Vf,trenchState]= SwarmSimFollowTrench(RobotParams, NRobot,SensorRange)
% SWARMSIMFOLLOWTRENCH - <Determines the vector Vf to follow in order for the swarm to follow a trench>

% Outputs:
%   trenchState      1=OnTrench, 2= offtrench, 3= offTrench_right
%                   4= offTrench_left 5= Not enough robots
%% Initialize Variables

% Determine number of robots based off length of robot params vector
N= floor(length(RobotParams)/4);

%% initialize variables
% position variables for 3-DOF omnibot are x,y,and theta (rotation about z-axis)
x=zeros(1,N);
y=zeros(1,N);
theta=zeros(1,N);
SensorValue= zeros(1,N); % this is the value of the "sensor reading" from each robot's "on-board" sensor.
d=zeros(1,N);
O=zeros(1,N);
O_minToRobots = zeros(1,N);
d_from_min=zeros(1,N);
delta_z_from_min=zeros(1,N);
amp=zeros(1,N);
% trenchVector=[0.0 0.0 0.0];
Vfx=0.0;
Vfy=0.0;
Vft=0.0;
Vf = [0.0 0.0 0.0];
trenchState = 0;



%% Set x,y,theta, and SensorValue inputs into an array

x(1,1:N)=RobotParams(1:4:end);
y(1,1:N)=RobotParams(2:4:end);
theta(1,1:N)=RobotParams(3:4:end);
SensorValue(1,1:N)=RobotParams(4:4:end);

% Determine distance and angle to each robot
d= sqrt( ( x(NRobot)-x ).^2 + ( y(NRobot)-y ).^2 );
O= atan2((y-y(NRobot)),(x-x(NRobot)));


%% Algorithm

inRange_idx=find(0<d & d<=SensorRange);

% Find robot with min sensor value
if length(inRange_idx) >=6
    min_robot_idx=find(SensorValue==min(SensorValue(inRange_idx)));
    if length(min_robot_idx)>1
        min_robot_idx = min_robot_idx(1);
    end
    TrenchBuffer = 0.1*(min(SensorValue(inRange_idx))-min(SensorValue(inRange_idx)));
    
    d_from_min = sqrt((x(min_robot_idx)*ones(size(x))-x ).^2 + ( y(min_robot_idx)*ones(size(x))-y).^2 );
    delta_z_from_min = SensorValue - SensorValue(min_robot_idx)*ones(size(SensorValue));
    amp= (d_from_min.^2)./delta_z_from_min;
    amp(delta_z_from_min<TrenchBuffer)=0;
    % Find max "amplitude" robot
    trench_robot_idxs=find(amp==max(amp(inRange_idx)));
    trench_robot_idx  = trench_robot_idxs(1); 
    [Vfx,Vfy,trenchState,R2rleft,R2rRight]= checktrench(min_robot_idx,trench_robot_idx,x,y,SensorValue,N,NRobot,inRange_idx);

else
    %Not enough robots inrange
    trenchState= 5;
    min_robot_idx = find(SensorValue== min(SensorValue(inRange_idx)));
    if length(min_robot_idx)>1
        min_robot_idx = min_robot_idx(1);
    end
    switch NRobot
        case min_robot_idx
            Vfx=0;
            Vfy=0;
        otherwise
            Vx=x(min_robot_idx)-x(NRobot);
            Vy=y(min_robot_idx)-y(NRobot);
            Vfx = Vx/sqrt(sum(Vx.^2+Vy.^2));
            Vfy = Vy/sqrt(sum(Vx.^2+Vy.^2));
    end
    
end
if trenchState ==2
    Vf = SwarmSimFindMax(RobotParams, NRobot, SensorRange);
    Vf = Vf./sqrt(sum(Vf.^2));
else
    Vf = [Vfx(1) Vfy(1) Vft(1)];
end
end
%% Helper Functions
function [Vfx,Vfy,trenchState,R2rleft,R2rRight]= checktrench(min_robot_idx,trench_robot_idx,x,y,SensorValue,N,NRobot,inRange_idx)
%Compute Homogeneous transform to look along vector from trenchRobot
%to min robot to determine number and curvature of robots on either
%side of proposed trench

%Find angles theta1 and theta2 for transform

theta1 = atan2(y(trench_robot_idx) - y(min_robot_idx),x(trench_robot_idx) - x(min_robot_idx));
theta2 = atan2(SensorValue(min_robot_idx)-SensorValue(trench_robot_idx),sqrt((y(trench_robot_idx) - y(min_robot_idx)).^2+(x(trench_robot_idx) - x(min_robot_idx))^2));
%Build Transforms
R1 = [cos(theta1), sin(theta1),0;-sin(theta1),cos(theta1),0;0 0 1 ;];
P01 = [x(min_robot_idx);y(min_robot_idx); SensorValue(min_robot_idx)];
H1 = [R1 -R1*P01; 0 0 0  1];
R2 = [cos(theta2),0, -sin(theta2);0 1 0; sin(theta2),0,cos(theta2)];
H2 = [R2 [0;0;0]; 0 0 0 1];
H = H2*H1;
%Convert coordinates
RPa = zeros(N*4,1);
RPa(1:4:end)=x;
RPa(2:4:end)=y;
RPa(3:4:end)= SensorValue;
RPa(4:4:end) = 1;
ACell = repmat({H}, 1, N);
HM = blkdiag(ACell{:});
RPap = HM*RPa;
Xsp = RPap(1:4:end);
Ysp = RPap(2:4:end);
SVp = RPap(3:4:end);
Xsp = Xsp(inRange_idx);
Ysp = Ysp(inRange_idx);
SVp = SVp(inRange_idx);


%Determine if robots are on a trench
nr_right = sum(Ysp<0);
nr_left = sum(Ysp>0);
nfr_right= sum(Ysp<0 & Xsp>=0); 
nfr_left = sum(Ysp>0 & Xsp>=0);

% Check if there aren't at least N_sided robots on one side
% Nsided must be at lest two
N_sided = floor(length(inRange_idx)/4);
if N_sided <2
    N_sided =2;
end
if nr_right <N_sided
    %Shift right
    MR = [x(trench_robot_idx)-x(min_robot_idx), y(trench_robot_idx)-y(min_robot_idx),0];
    V = cross(MR ,[0,0,1]);
    Vfs = V(1:2);
    Vm = sqrt(sum(Vfs.^2));
    Vfx=Vfs(1)/Vm;
    Vfy=Vfs(2)/Vm;
    trenchState= 4;
    R2rleft =0;
    R2rRight=0;
elseif nr_left < N_sided
    %Shift left
    MR = [x(trench_robot_idx)-x(min_robot_idx), y(trench_robot_idx)-y(min_robot_idx),0];
    V = cross(MR ,[0,0,-1]);
    Vfs = V(1:2);
    Vm = sqrt(sum(Vfs.^2));
    Vfx=Vfs(1)/Vm;
    Vfy=Vfs(2)/Vm;
    trenchState= 3;
    R2rleft =0;
    R2rRight=0;
else
    %There are at least two robots on either side of the trench
    R2Limit= 0.5;
    if nfr_right >=2 && nfr_left>=2
        %Calculate slope of robots on either side of the trench and R^2 values
        [Prr,~,R2rRight] = fastPolyfit(Ysp(Ysp<=0 & Xsp>=0),SVp(Ysp<=0 & Xsp>=0));
        [Prl,~,R2rleft] = fastPolyfit(Ysp(Ysp>=0),SVp(Ysp>=0));
    else
        R2rleft =0;
        R2rRight=0;
        Prl =0;
        Prr=0;
    end
    
    if Prl < 0 && Prr>0 && R2rleft>=R2Limit && R2rRight >= R2Limit
        %Trenchm with sufficient number of robots on either side
        Vfs = [x(trench_robot_idx)-x(min_robot_idx), y(trench_robot_idx)-y(min_robot_idx)];
        Vm = sqrt(sum(Vfs.^2));
        Vfx=Vfs(1)/Vm;
        Vfy=Vfs(2)/Vm;
        trenchState=1;
    elseif Prl < 0 && Prr>0 && R2rleft< R2Limit && R2rRight >= R2Limit
        %Shift left
        MR = [x(trench_robot_idx)-x(min_robot_idx), y(trench_robot_idx)-y(min_robot_idx),0];
        V = cross(MR ,[0,0,-1]);
        Vfs = V(1:2);
        Vm = sqrt(sum(Vfs.^2));
        Vfx=Vfs(1)/Vm;
        Vfy=Vfs(2)/Vm;
        trenchState= 3;
    elseif Prl < 0 && Prr>0 && R2rleft>= R2Limit && R2rRight < R2Limit
        %Shift right
        MR = [x(trench_robot_idx)-x(min_robot_idx), y(trench_robot_idx)-y(min_robot_idx),0];
        V = cross(MR ,[0,0,1]);
        Vfs = V(1:2);
        Vm = sqrt(sum(Vfs.^2));
        Vfx=Vfs(1)/Vm;
        Vfy=Vfs(2)/Vm;
        trenchState= 4;
    else
        %Either slopes are not correct or lines fit too poorly
        %Not a trench so find min
        trenchState= 2;

        Vfx = 0;
        Vfy = 0; 
    end
end

end

%% SUBFUNCTIONS
function [k,b,r] = fastPolyfit(x,y)
%https://stackoverflow.com/questions/48802871/fast-fit-of-linear-function-matlab
sumX=nansum(x);
sumY=nansum(y);
sumX2=nansum(x.^2);
sum2X=sumX.^2;
sum2Y=sumY.^2;
N = length(x);

XY=x.*y;
sumXY=nansum(XY);
numerator=N.*sumXY-sumX.*sumY;
denominator=(N.*sumX2-sum2X);

k=numerator./denominator;
b=(sumY-k.*sumX)./N;
r=abs(numerator./sqrt(denominator.*(N.*nansum(y.^2)-sum2Y)));
end