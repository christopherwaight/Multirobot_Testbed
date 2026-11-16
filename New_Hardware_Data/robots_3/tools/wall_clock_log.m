function y = wall_clock_log(t)
% LOGGING ONLY: the code inside the root 'LOG timing fn' MATLAB Function block.
% Input: simulation time from a Digital Clock. Output (1x2): [sim_time, wall_clock_posix_seconds].
% diff(timing(:,2)) is the real time each step took; compare with the fixed step (0.1 s).
coder.extrinsic('posixtime', 'datetime');
w = 0;
w = posixtime(datetime('now', 'TimeZone', 'local'));
y = [t, w];
end
