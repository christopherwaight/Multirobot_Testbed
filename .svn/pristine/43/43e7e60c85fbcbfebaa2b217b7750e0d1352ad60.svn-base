function [] = SVProperties(data,Num_Robots,SVDesired)
figure
hold on
for j = 1:length(data)
    e = zeros(size(data(1,j).robot(1).sensor_value)); 
    for i = 1:Num_Robots
        e = e + (data(1,j).robot(i).sensor_value - SVDesired).^2; 
    end
    e_rms = sqrt(e./Num_Robots); 
    plot(data(1,j).robot(1).time, e_rms)
end
xlabel('Time (s)')
ylabel('e_{rms}')
title('RMS Error to Desired Value')
box on 
hold off
end


