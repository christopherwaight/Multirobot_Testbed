function [] = OrbitProperties(data,Num_Robots)
g = ceil(sqrt(numel(data)));
figure
for j = 1:length(data)
    % Create square grid of subplots for all parallel simulations
    P = subplot(g,g,j);
    hold on
    Xs = []; 
    for i = 1:Num_Robots
        Xs = [Xs; [data(1,j).robot(i).x(1:10:end),data(1,j).robot(i).y(1:10:end)]];
    end
    [A , c] = MinVolEllipse(Xs', 0.001);
    if Num_Robots>2
        Ellipse_plot(A, c)
        scatter(Xs(:,1), Xs(:,2))
        axis equal
        box on
        hold off
        [~, D, ~] = svd(A);

        % get the major and minor axes
        %------------------------------------
        a(j) = 1/sqrt(D(1,1));
        b(j) = 1/sqrt(D(2,2));
    end
    %% Calculate max distance for simulation
end
if Num_Robots>2
    disp('The minor axes were:')
    disp(a)
    disp('The major axes were:')
    disp(b)
    disp('Min and max minor axis were:')
    disp(min(a))
    disp(max(a))
    disp('Min and max major axis were:')
    disp(min(b))
    disp(max(b))
    disp('Average minor axis and average major axis were')
    disp(mean(a))
    disp(mean(b))
    disp('Standard dev for minor axis and major axis were')
    disp(std(a))
    disp(std(b))
    e = sqrt(1 - (a./b).^2); 
    disp('All eccentricities:')
    disp(e); 
    disp('Min and max eccentricity')
    disp(min(e)); 
    disp(max(e)); 
    disp('Average eccentricity:')
    disp(mean(e));
end
end


