function [everything_hue_net, everything_sat_net, tr1, tr2] = trainNetworks(everything_hue_net, everything_sat_net, inputs_normalized, hue_targets_sin_normalized, hue_targets_cos_normalized, sat_targets_normalized)



    % Train the hue network (outputs both sin and cos of hue)
    fprintf('\nTraining Networks...\n');
    [everything_hue_net, tr1] = train(everything_hue_net, inputs_normalized, ...
                                    [hue_targets_sin_normalized; hue_targets_cos_normalized]);
    
    % Train the saturation network

    [everything_sat_net, tr2] = train(everything_sat_net, inputs_normalized, sat_targets_normalized);
end