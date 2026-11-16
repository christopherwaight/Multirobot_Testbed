function report = clean_campaign_files(srcDir, dstDir, doWrite)
%CLEAN_CAMPAIGN_FILES  Make clean copies of hardware run files that contain ONLY what the run logged.
%
%   clean_campaign_files(srcDir)                  dry run: report what would be kept and dropped
%   clean_campaign_files(srcDir, dstDir, true)    write cleaned copies to dstDir (same sub-folders)
%
% Why: runs are saved with "save" of the whole workspace, so each file also carries STALE variables
% from earlier sessions (2025 old-name variables, 4-robot leftovers in 3-robot files and vice versa).
% See README_hardware.md section 6, "Stale variables".
%
% Rules:
%  - The ORIGINAL files are never modified. Cleaned copies go to dstDir.
%  - The robot count comes from the run's own command log, cmd_robots_selected (3 columns per robot).
%  - Kept = the variables that model logs (its run_config.log_legend) + tout + run_config.
%    The 4-robot model's debug FK_results (assignin, capped at 201 rows) is kept for 4-robot files only.
%  - Every kept time series must have exactly numel(tout) rows; otherwise the file is reported and
%    NOT written.
%  - Any expected variable missing from a file is reported (and the file is not written).
%  - Files without run_config (pre-2026-11-12 logging) and non-run files are skipped and listed.
%
% Written by Claude (AI assistant) on 2026-11-15.

if nargin < 2, dstDir = ''; end
if nargin < 3, doWrite = false; end
if doWrite && isempty(dstDir), error('give dstDir to write'); end
if doWrite && strcmpi(strip(fullfile(srcDir), 'right', filesep), strip(fullfile(dstDir), 'right', filesep))
    error('dstDir must differ from srcDir: originals are never overwritten');
end

keep3 = [{'pose_r1', 'pose_r2', 'pose_r3'}, ...
    strcat('sensor_rgbk_raw_r', {'1', '2', '3'}), strcat('sensor_rgbk_norm_r', {'1', '2', '3'}), ...
    {'sensor_hue', 'sensor_sat', 'field_uv', 'field_jacobian', 'field_cp', 'orbit_radius', ...
     'cluster_state', 'setpoint', 'err_setpoint', 'err_jacobian'}, ...
    strcat('cmd_cluster_', {'setpoint', 'jacobian', 'vectorsum', 'ascent', 'descent'}), ...
    strcat('cmd_robots_', {'setpoint', 'jacobian', 'vectorsum', 'ascent', 'descent'}), ...
    {'cmd_robots_selected'}, strcat('cmd_body_r', {'1', '2', '3'}), {'simfield_hue', 'simfield_sat', 'timing'}];
keep4 = [strcat('pose_r', {'1', '2', '3', '4'}), ...
    strcat('sensor_rgbk_raw_r', {'1', '2', '3', '4'}), strcat('sensor_rgbk_norm_r', {'1', '2', '3', '4'}), ...
    {'sensor_hue', 'sensor_sat', 'field_uv', 'field_jacobian', 'field_cp_jacobian', 'field_cp_jacobians', ...
     'orbit_radius_jacobian', 'orbit_radius_jacobians', 'cluster_state', 'setpoint', ...
     'err_setpoint', 'err_jacobian', 'err_jacobians'}, ...
    strcat('cmd_cluster_', {'setpoint', 'jacobian', 'jacobians'}), ...
    strcat('cmd_robots_', {'setpoint', 'jacobian', 'jacobians'}), ...
    {'cmd_robots_selected'}, strcat('cmd_body_r', {'1', '2', '3', '4'}), {'diag_angle_error_jacobians', 'timing'}];
notTimeSeries = {'run_config', 'tout', 'FK_results'};

fs = dir(fullfile(srcDir, '**', '*.mat'));
report = struct('file', {}, 'robots', {}, 'kept', {}, 'dropped', {}, 'status', {});
nW = 0; nSkip = 0; nBad = 0;
for i = 1:numel(fs)
    src = fullfile(fs(i).folder, fs(i).name);
    rel = strrep(src, [fullfile(srcDir) filesep], '');
    S = load(src);
    if ~isfield(S, 'run_config') || ~isfield(S, 'cmd_robots_selected') || ~isfield(S, 'tout')
        report(end+1) = struct('file', rel, 'robots', NaN, 'kept', {{}}, 'dropped', {{}}, 'status', 'SKIPPED (not a new-logging run file)'); %#ok<AGROW>
        nSkip = nSkip + 1; continue;
    end
    nr = size(S.cmd_robots_selected, 2) / 3;
    if nr == 3, want = keep3; elseif nr == 4, want = [keep4 {'FK_results'}]; else, want = {}; end
    want = [want, {'tout', 'run_config'}];
    v = fieldnames(S)';
    kept = intersect(want, v, 'stable'); dropped = setdiff(v, kept, 'stable');
    missing = setdiff(setdiff(want, {'FK_results'}), v);
    n = numel(S.tout); badLen = {};
    for k = 1:numel(kept)
        if any(strcmp(kept{k}, notTimeSeries)), continue; end
        if size(S.(kept{k}), 1) ~= n, badLen{end+1} = sprintf('%s(%d rows)', kept{k}, size(S.(kept{k}), 1)); end %#ok<AGROW>
    end
    if ~isempty(missing) || ~isempty(badLen) || ~ismember(nr, [3 4])
        st = sprintf('NOT CLEAN: missing [%s] wrong length [%s]', strjoin(missing, ' '), strjoin(badLen, ' '));
        nBad = nBad + 1;
    else
        st = 'ok';
        if doWrite
            dst = fullfile(dstDir, rel);
            if ~isfolder(fileparts(dst)), mkdir(fileparts(dst)); end
            C = rmfield(S, dropped); %#ok<NASGU>
            save(dst, '-struct', 'C', '-v7');
            % verify the copy: same kept variables, bit-identical
            T = load(dst);
            if ~isequal(sort(fieldnames(T)), sort(kept')) || any(cellfun(@(f) ~isequaln(T.(f), S.(f)), kept))
                error('verification failed for %s', dst);
            end
            st = 'written + verified'; nW = nW + 1;
        end
    end
    report(end+1) = struct('file', rel, 'robots', nr, 'kept', {kept}, 'dropped', {dropped}, 'status', st); %#ok<AGROW>
end

% summary, grouped by the set of dropped variables
fprintf('%d files scanned | %s | %d skipped | %d not clean\n', numel(fs), ...
    ternary(doWrite, sprintf('%d written', nW), 'DRY RUN, nothing written'), nSkip, nBad);
ok = report(~isnan([report.robots]));
keys = cellfun(@(d) strjoin(sort(d), ' '), {ok.dropped}, 'uni', 0);
[u, ~, g] = unique(keys);
for j = 1:numel(u)
    m = ok(g == j);
    fprintf('\n%d files (%s robots) drop %d variables: %s\n   e.g. %s\n', numel(m), num2str(unique([m.robots])), numel(m(1).dropped), u{j}, m(1).file);
end
fprintf('\n');
for j = 1:numel(report)
    if ~strcmp(report(j).status, 'ok') && ~startsWith(report(j).status, 'written'), fprintf('%s : %s\n', report(j).file, report(j).status); end
end
if doWrite
    T = struct2table(rmfield(report, {'kept'}), 'AsArray', true);
    T.dropped = cellfun(@(d) strjoin(d, ' '), T.dropped, 'uni', 0);
    writetable(T, fullfile(dstDir, 'CLEANING_MANIFEST.csv'));
end
end
function o = ternary(c, a, b), if c, o = a; else, o = b; end, end
