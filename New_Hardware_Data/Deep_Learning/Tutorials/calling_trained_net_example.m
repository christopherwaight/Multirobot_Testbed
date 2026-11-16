% ... (Your training code as before) ...

% Save the trained network
save('trainedNetwork.mat', 'net');




% Then in a simulink block, make this function
function output = myNeuralNetworkFunction(input, netID)
    % Use persistent variables to store each network
    persistent loadedNet1 loadedNet2 loadedNet3;

    if netID == 1
        if isempty(loadedNet1)
            loadedNet1 = evalin('base', 'net1');
            disp('Network 1 loaded!');
        end
        output = loadedNet1(input);
    elseif netID == 2
        if isempty(loadedNet2)
            loadedNet2 = evalin('base', 'net2');
            disp('Network 2 loaded!');
        end
        output = loadedNet2(input);
    elseif netID == 3
        if isempty(loadedNet3)
            loadedNet3 = evalin('base', 'net3');
            disp('Network 3 loaded!');
        end
        output = loadedNet3(input);
    else
        error('Invalid network ID');
    end
end