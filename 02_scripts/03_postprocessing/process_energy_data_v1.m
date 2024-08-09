close all
%clear all

% This script takes the output variables of various energies and plots them if they are non-zero

% Get from shell-script:
%jobname = 'Seed03_v4';
%inc = 10;

% Test if jobname is correct
disp(['Jobname is: ', jobname])

folder = append(jobname, "_results");
global resultfiles
resultfiles = './' + folder + '/';

global datafolder
datafolder = resultfiles + 'outputdata/';

% List of energy variables
energy_vars = {
    'ALLAE', 'ALLCCDW', 'ALLCCE', 'ALLCCEN', 'ALLCCET', ...
    'ALLCCSD', 'ALLCCSDN', 'ALLCCSDT', 'ALLCD', 'ALLDMD', ...
    'ALLDTI', 'ALLEE', 'ALLQB', 'ALLWK', 'ALLFD', ...
    'ALLIE', 'ALLJD', 'ALLKE', 'ALLKL', 'ALLPD', ...
    'ALLSD', 'ALLSE', 'ETOTAL', 'ALLVD'
};

% Read time data
time_data = read_data('time');

% Threshold to filter out nonsensical values
threshold = 1e6;

jobname_underscore = regexprep(jobname, '[\\\^\_]','\\$0');

% Loop through each energy variable, read the data, and plot if non-zero
for i = 1:length(energy_vars)
    energy_var = energy_vars{i};
    energy_data = read_data(energy_var);

    % Filter out nonsensical values
    energy_data_filtered = energy_data(energy_data < threshold);

    % Check if the energy data is non-zero
    if any(energy_data_filtered)
        figure;
        plot(time_data(1:length(energy_data_filtered)), energy_data_filtered, 'DisplayName', energy_var, 'LineWidth', 1);
        hold on;
        title(strcat(jobname_underscore, ' - ', energy_var));
        xlabel('Time');
        ylabel('Energy');
        legend('Location', 'eastoutside');
        
        % Save plots
        savefig(resultfiles + energy_var);
        saveas(gcf, resultfiles + energy_var + '.png');
        saveas(gcf, resultfiles + energy_var + '.svg');
        hold off;
    else
        disp([energy_var ' is all zeros or contains only nonsensical values and will not be plotted.']);
    end
end

% Define a list of line styles
line_styles = {'-', '--', ':', '-.'};
line_colors = lines(length(energy_vars)); % Get a colormap with a unique color for each variable

% Initialize a new figure for the combined plot
figure;
hold on;
title(strcat(jobname_underscore, ' - Combined Energy Plot'));
xlabel('Time');
ylabel('Energy');
legend_entries = {};

% Loop through each energy variable, read the data, filter nonsensical values, and plot if non-zero
for i = 1:length(energy_vars)
    energy_var = energy_vars{i};
    energy_data = read_data(energy_var);

    % Filter out nonsensical values
    energy_data_filtered = energy_data(energy_data < threshold);

    % Check if the energy data is non-zero
    if any(energy_data_filtered)
        % Determine the line style and color
        line_style = line_styles{mod(i-1, length(line_styles)) + 1};
        line_color = line_colors(i, :);

        % Plot the filtered data on the combined plot
        plot(time_data(1:length(energy_data_filtered)), energy_data_filtered, 'DisplayName', energy_var, 'LineWidth', 1, 'LineStyle', line_style, 'Color', line_color);

        % Annotate the line with a number
        text(time_data(end), energy_data_filtered(end), num2str(i), 'FontSize', 6, 'Color', line_color);

        % Add the variable name with its number to the legend entries
        legend_entries{end+1} = [num2str(i) ' ' energy_var];
    else
        disp([energy_var ' is all zeros or contains only nonsensical values and will not be plotted.']);
    end
end

% Add the legend to the combined plot
legend(legend_entries, 'Location', 'eastoutside');

% Save the combined plot
savefig(resultfiles + 'Combined_Energy_Plot');
saveas(gcf, resultfiles + 'Combined_Energy_Plot.png');
saveas(gcf, resultfiles + 'Combined_Energy_Plot.svg');

hold on

%%%%%%%%%%%%%%%%%% include increments as vertical lines %%%%%%%%%%%%%%%%%
inc_1 = inc+1;

disp_X = time_data(inc_1:inc:end);
incs = (1:length(time_data))';
incs_X = incs(inc:inc:end);
incs_X = num2str(incs_X);

for i = 1:length(disp_X)
    incnum = incs_X(i,:);
    incline = xline(disp_X(i),'--r',incnum,'HandleVisibility','off');
    incline.LabelHorizontalAlignment = 'center';
end

hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inc_str = num2str(inc);

savefig(resultfiles + 'Combined_Energy_Plot' + inc_str)
saveas(gcf, resultfiles + 'Combined_Energy_Plot' + inc_str + '.png')
saveas(gcf, resultfiles + 'Combined_Energy_Plot' + inc_str + '.svg')


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% write energy report %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create Energy_report.txt file
report_filename = fullfile(resultfiles, 'Energy_report.txt');
fid = fopen(report_filename, 'w');

% Write initial information
fprintf(fid, 'In the analysis [%s.inp] the following keyword-specific stabilization terms are used:\n', jobname);

% Read the jobname input file
input_filename = [jobname, '.inp'];
input_file = fopen(input_filename, 'r');
input_lines = textscan(input_file, '%s', 'Delimiter', '\n');
fclose(input_file);

% Search for Material and Step input files in the jobname input file
material_file = '';
step_file = '';
input_lines = input_lines{1};  % Extract the cell array inside the main cell

for i = 1:length(input_lines)
    line = input_lines{i};
    if contains(line, '*Include, Input=')
        % Use regular expression to extract the file path
        matches = regexp(line, 'Input=(.*)', 'tokens');
        if ~isempty(matches)
            filepath = strtrim(matches{1}{1});
            if contains(filepath, 'Material.inp')
                material_file = filepath;
            elseif contains(filepath, 'Step')
                step_file = filepath;
            end
        end
    end
end

% Extract stabilization terms from the Material input file
if ~isempty(material_file)
    fprintf(fid, '\nIn %s:\n', material_file);
    material_filepath = fullfile(fileparts(input_filename), material_file);
    material_file = fopen(material_filepath, 'r');
    material_lines = textscan(material_file, '%s', 'Delimiter', '\n');
    fclose(material_file);
    
    material_lines = material_lines{1};  % Extract the cell array inside the main cell
    for i = 1:length(material_lines)
        line = material_lines{i};
        if contains(line, '*Damage Stabilization')
            fprintf(fid, '%s\n', line);
            if i + 1 <= length(material_lines)
                fprintf(fid, '%s\n', material_lines{i+1}); % Assuming the value is on the next line
            end
        end
    end
end

% Extract stabilization terms from the Step input file
if ~isempty(step_file)
    fprintf(fid, '\nIn %s:\n', step_file);
    step_filepath = fullfile(fileparts(input_filename), step_file);
    step_file = fopen(step_filepath, 'r');
    step_lines = textscan(step_file, '%s', 'Delimiter', '\n');
    fclose(step_file);
    
    step_lines = step_lines{1};  % Extract the cell array inside the main cell
    for i = 1:length(step_lines)
        line = step_lines{i};
        if contains(line, '*Static, stabilize')
            fprintf(fid, '%s\n', line);
            if i + 1 <= length(step_lines)
                fprintf(fid, '%s\n', step_lines{i+1}); % Assuming the values are on the next line
            end
        elseif contains(line, '*Contact')
            % Check for Contact stabilization block
            if i + 1 <= length(step_lines) && contains(step_lines{i+1}, '*Contact Stabilization')
                fprintf(fid, '%s\n', line);
                fprintf(fid, '%s\n', step_lines{i+1});
                if i + 2 <= length(step_lines)
                    fprintf(fid, '%s\n', step_lines{i+2}); % Assuming the values are on the next line
                end
            end
        end
    end
end

% Close the report file
fclose(fid);


% Define percentages for the analysis end times
percentages = [0, 0.01, 0.10, 0.50, 0.90, 0.95, 0.99, 1.00];

% Read time data
time_data = read_data('time');

% Calculate the target times for each percentage
analysis_end_time = time_data(end);
target_times = analysis_end_time * percentages;

% Initialize a matrix to store energy values
num_percentages = length(percentages);
energy_files = dir(fullfile(datafolder, '*.txt'));
num_energy_terms = length(energy_files);

% Read energy data from files
energy_data = cell(num_energy_terms, 1);
energy_names = cell(num_energy_terms, 1);
for i = 1:num_energy_terms
    energy_names{i} = strrep(energy_files(i).name, '.txt', '');
    energy_data{i} = read_data(energy_names{i});
end

% Find the closest indices for each target time
closest_indices = arrayfun(@(t) find(abs(time_data - t) == min(abs(time_data - t)), 1), target_times);

% Calculate energy ratios for specific energy terms relative to ALLIE
allie_data = read_data('ALLIE');
energy_ratios = nan(num_energy_terms, num_percentages);
for i = 1:num_energy_terms
    for j = 1:num_percentages
        if strcmp(energy_names{i}, 'ALLIE')
            energy_ratios(i, j) = energy_data{i}(closest_indices(j));
        else
            energy_ratios(i, j) = (energy_data{i}(closest_indices(j)) / allie_data(closest_indices(j))) * 100;
        end
    end
end

% Prepare the table for Energy_report.txt
report_filename = fullfile(resultfiles, 'Energy_report.txt');
fid = fopen(report_filename, 'a'); % Append to the existing file

fprintf(fid, '\n');
fprintf(fid, '%% till analysis end   |');
fprintf(fid, '%10.0f%%|', percentages(2:end) * 100);
fprintf(fid, '\n');
fprintf(fid, '-----------------------------------------------------------------------------------------------------------\n');

fprintf(fid, 'steptime              |');
fprintf(fid, '%11.4f|', target_times(2:end));
fprintf(fid, '\n');

fprintf(fid, 'inc number            |');
fprintf(fid, '%11d|', closest_indices(2:end));
fprintf(fid, '\n');
fprintf(fid, '-----------------------------------------------------------------------------------------------------------\n');

% Print energy ratios
energy_ratio_names = {'ALLCD', 'ALLSD', 'ALLVD', 'ALLPD', 'ETOTAL'};
for i = 1:num_energy_terms
    if any(strcmp(energy_names{i}, energy_ratio_names))
        fprintf(fid, '%-22s|', [energy_names{i}, '/ALLIE*100']);
        fprintf(fid, '%11.4f|', energy_ratios(i, 2:end));
        fprintf(fid, '\n');
    end
end
fprintf(fid, '-----------------------------------------------------------------------------------------------------------\n');

% Print absolute energy terms with special handling for zero values
for i = 1:num_energy_terms
    fprintf(fid, '%-22s|', energy_names{i});
    for j = 2:num_percentages
        value = energy_data{i}(closest_indices(j));
        if value == 0
            fprintf(fid, '%11.1f|', 0.0);
        else
            fprintf(fid, '%11.4e|', value);
        end
    end
    fprintf(fid, '\n');
end

fclose(fid);

%%%% Function to read data
function data = read_data(var_name)
    global datafolder
    file_name_long = fullfile(datafolder, [var_name, '.txt']);
    file = fopen(file_name_long, 'r');
    data = textscan(file, '%s');
    fclose(file);
    data = str2double(data{1}(1:1:end));
end

