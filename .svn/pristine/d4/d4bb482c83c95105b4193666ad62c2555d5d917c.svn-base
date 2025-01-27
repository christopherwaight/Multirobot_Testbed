function [] = FormationProperties(data,Num_Robots,SimParams)
g = ceil(sqrt(numel(data)));
f_formPE = figure;
p_formPE = uipanel('Parent',f_formPE,'BorderType','none');
p_formPE.Title = 'Formation Percent Error';
p_formPE.TitlePosition = 'centertop';
p_formPE.FontSize = 12;
p_formPE.FontWeight = 'bold';
f_formPEc = figure;

for k = 1:length(data)
    % Create square grid of subplots for all parallel simulations
    figure(f_formPE)
    P = subplot(g,g,k,'Parent',p_formPE);
    hold on
    maxe = nan(size(data(k).robot(1).x));
    for i = 1:Num_Robots
        for j = (i+1):Num_Robots
            dij = sqrt((data(k).robot(i).x - data(k).robot(j).x).^2 + (data(k).robot(i).y - data(k).robot(j).y).^2);
            eij = (dij - SimParams.Distances(i,j))./ SimParams.Distances(i,j);
            maxe = max(abs(eij), maxe);
            plot(data(k).robot(i).time, eij);
        end
    end
    box on
    grid on
    xlabel('Time (s)')
    ylabel('PE (%)')
    hold off
    figure(f_formPEc)
    hold on
    plot(data(k).robot(i).time,maxe)
    hold off
    
    
end
figure(f_formPEc)
hold on
grid on
xlabel('Time (s)')
ylabel('Max Percent Error (%)')
title('Formation Control Max Percent Error')
box on
hold off


end


