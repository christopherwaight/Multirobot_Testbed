function [] = plotScalarValueTimeHistory(Robot_Data, f,DesiredValue,behavior,Nrobot, yl)

g = ceil(sqrt(numel(Robot_Data)));
f_svh = figure;
p_svh = uipanel('Parent',f_svh,'BorderType','none'); 
p_svh.Title = 'Time History of Scalar Values'; 
p_svh.TitlePosition = 'centertop'; 
p_svh.FontSize = 12;
p_svh.FontWeight = 'bold';

f_svmh  =figure;
for k = 1:numel(Robot_Data)
    aggSV = nan(size(Robot_Data(1).robot(1).sensor_value));
    figure(f_svh);
    P = subplot(g,g,k,'Parent',p_svh);
    hold on
    for i= 1:Nrobot
        plot(Robot_Data(k).robot(i).time,Robot_Data(k).robot(i).sensor_value)
        aggSV = f(aggSV, Robot_Data(k).robot(i).sensor_value);
    end
    if any(strcmp(behavior,'Find Max')) 
        ps = plot(Robot_Data(1).robot(1).time,ones(size(Robot_Data(1).robot(1).sensor_value))*DesiredValue,'k--');
        legend(ps, 'Known Global Maximum');
    elseif any(strcmp(behavior,'Find Min')) 
        ps = plot(Robot_Data(1).robot(1).time,ones(size(Robot_Data(1).robot(1).sensor_value))*DesiredValue,'k--');
        legend(ps, 'Known Global Minimum'); % 
    elseif any(strcmp(behavior,'Contour Following')) 
        ps = plot(Robot_Data(1).robot(1).time,ones(size(Robot_Data(1).robot(1).sensor_value))*DesiredValue,'k--');
        legend(ps, 'Desired Value'); % 
    end
    
    xlabel('Time (s)','fontsize',12)
    ylabel('Sensor Value','fontsize',12)
    box on
    hold off
    figure(f_svmh);
    hold on
    if any(strcmp(behavior,'Contour Following'))
        plot(Robot_Data(k).robot(i).time, aggSV./Nrobot)
    else
        plot(Robot_Data(k).robot(i).time, aggSV);
    end
    box on
    hold off
    
end
figure(f_svmh);
hold on
xlabel('Time (s)')
ylabel(yl)
title('Aggregate Time History of Sensor Values')
hold off
