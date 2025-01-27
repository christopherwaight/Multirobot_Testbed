function [] = TrajectorytProperties(data,Num_Robots,trajFun)
g = ceil(sqrt(numel(data)));
f_traj = figure;
f_ec = figure;
p_traj = uipanel('Parent',f_traj,'BorderType','none'); 
p_traj.Title = 'COM Trajectory'; 
p_traj.TitlePosition = 'centertop'; 
p_traj.FontSize = 12;
p_traj.FontWeight = 'bold';

time = data(1).robot(1).time;
[xd, yd] = trajFun(time); 
for k = 1:length(data)
    % Create square grid of subplots for all parallel simulations
    x_PI= zeros(length(data(k).robot(1).x),Num_Robots);
    y_PI= zeros(length(data(k).robot(1).y),Num_Robots);
    theta_PI= zeros(length(data(k).robot(1).theta),Num_Robots);
    sensor_value_PI = zeros(length(data(k).robot(1).sensor_value),Num_Robots);
    
    for i=1:Num_Robots
        x_PI(:,i) = data(k).robot(i).x;
        y_PI(:,i) = data(k).robot(i).y;
        theta_PI(:,i)= data(k).robot(i).theta;
        sensor_value_PI(:,i)= data(k).robot(i).sensor_value;
    end
    
    figure(f_traj)
    P = subplot(g,g,k,'Parent',p_traj);
    hold on
    xcom = sum(x_PI,2)/Num_Robots;
    ycom = sum(y_PI,2)/Num_Robots;
    

    dt = 1;
    plot(xd(1:dt:end),yd(1:dt:end), '-')
    plot(xcom(1:dt:end),ycom(1:dt:end),'.-.')
    box on
    xlabel('x(m)')
    ylabel('y(m)')
    legend('Desired Trajectory','COM Trajectory')
    hold off
    
    %calculate trajectory disturbance magnitude
    trajdistmag = sqrt((xd-xcom).^2 + (yd - ycom).^2);
    figure(f_ec)
    hold on
    plot(data(k).robot(1).time, trajdistmag)
    hold off
    
end
    figure(f_ec)
    xlabel('Time (s)')
    ylabel('Disturbance Magnitude (m)')
    title('Trajectory Following Disturbance Magnitude')
    box on
    
end




