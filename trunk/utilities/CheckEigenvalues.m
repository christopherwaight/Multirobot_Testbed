figure
idx =1; 
for i = 1:10:length(time)
    N = length(Robot_Data.robot);
    A = zeros(2*N,2*N);
    x_state= zeros(2*N,1);
    x_state(1:2:end) = x_PI(i,:); 
    x_state(2:2:end) = y_PI(i,:);
    RobotParams = zeros(4*N,1);
    RobotParams(1:4:end) = x_PI(i,:); 
    RobotParams(2:4:end) = y_PI(i,:);
    RobotParams(4:4:end) = sensor_value_PI(i,:);
    for j = 1:length(Robot_Data.robot)
        d(1:2:2*N) = sqrt((x_state(1:2:end)-x_state(2*j-1)).^2 + (x_state(2:2:end)-x_state(2*j)).^2); 
        d(2:2:2*N) = sqrt((x_state(1:2:end)-x_state(2*j-1)).^2 + (x_state(2:2:end)-x_state(2*j)).^2); 
        d(d>SimParams.SENSOR_RANGE| d==0)=inf;
        D(idx,j) = sum(d(1:2:2*N)<=0.6);
        P = diag(1./d);

        %% Attract Behavior 

        % Determine distance and angle to each robot 
        L = eye(2*N);
        L(1:2:end,2*j-1) = L(1:2:2*N,2*j-1)-1;
        L(2:2:end,2*j) = L(2:2:2*N,2*j)-1;
        %W = distRelativeToSetpoint(j,RobotParams,SimParams.Distances);
        W= OnesWeights(j,RobotParams,{});
        %W = GradientAscent(j,RobotParams,300);
        %A(2*j-1:2*j,:) = W*P*L; 
        A(2*j-1:2*j,:) = [0 -1; 1 0]*W*P*L; 
    end
    hold on
    [E,V] = eig(A);
%     for k = 1:length(E)
%         %figure
%         ex = E(1:2:end,k); 
%         ey = E(2:2:end,k); 
%         %scatter(ex,ey)
%     end
    %pause(0.01)
    hold off
    e(:,idx) = eig(A);
    er(:,idx) = sort(real(e(:,idx))); 
    ei(:,idx) = sort(imag(e(:,idx))); 
    idx = idx+1; 
end
    figure
    plot(time(1:10:end),er)
    title('Eigenvalues: Real Part')
    figure
    plot(time(1:10:end),ei)
    title('Eigenvalues: Imaginary Part')
    figure
    plot(time(1:10:end),D)
    title('obsavoid triggered')
    
    %% 
    ei1 = ei(:,1); 
    unique(ei1(ei1>0))