close all

%run locally, not via bash...
%root_dir = 'C:\Users\Senk\Desktop\Droplet_Tests_FEA\01_neural_network_project\01_data\parameter_files\geometrical_samples\v2';
%root_dir = 'C:\Users\Senk\Desktop\Droplet_Tests_FEA\01_neural_network_project\01_data\parameter_files\mechanical_samples\v2';

% Get the result folders inside the numbered subdirectories
result_folders = dir(fullfile(root_dir, '**', '*_results'));

% Create a directory for comparison results if it doesn't exist
compare_folder = fullfile(root_dir, '_compare_allresults');
if ~exist(compare_folder, 'dir')
    mkdir(compare_folder);
end

% Define a list of line styles
line_styles = {'-', '--', ':', '-.'};
line_colors = lines(length(result_folders)); % Get a colormap with a unique color for each variable

% Plot and save original data
figure;
hold on;
title('Force-Displacement Comparison');
xlabel('Displacement (mm)');
ylabel('Force (N)');
legend_entries = {};

for i = 1:length(result_folders)
    folder_name = result_folders(i).name;
    folder_path = fullfile(result_folders(i).folder, folder_name);  % Correct the full path
    data_folder = fullfile(folder_path, 'outputdata');  % Now correctly points to the 'outputdata' directory

    % Specific RF2 and U2 filenames
    rf2_file_name = 'RF2_BC3.txt';
    u2_file_name = 'U2_BC3.txt';

    % Read data from files
    rf2_data = read_data(fullfile(data_folder, rf2_file_name));
    u2_data = read_data(fullfile(data_folder, u2_file_name));

    % Determine the line style and color
    line_style = line_styles{mod(i-1, length(line_styles)) + 1};
    line_color = line_colors(i, :);

    % Plot the original data
    plot(u2_data, rf2_data, 'LineWidth', 1, 'LineStyle', line_style, 'Color', line_color);

    % Annotate the line with a number
    text(u2_data(end), rf2_data(end), num2str(i), 'FontSize', 6, 'Color', line_color);

    % Add legend entry with escaped underscores
    legend_entries{end+1} = [num2str(i) ' ' strrep(strrep(folder_name, '_results', ''), '_', '\_')];
end

% Add the legend to the original comparison plot
legend(legend_entries, 'Location', 'eastoutside');

% Save the original comparison plot
savefig(fullfile(compare_folder, 'Force-Displacement_Comparison'));
saveas(gcf, fullfile(compare_folder, 'Force-Displacement_Comparison.png'));
saveas(gcf, fullfile(compare_folder, 'Force-Displacement_Comparison.svg'));

hold off;

% Plot and save adjusted data
figure;
hold on;
title('Force-Displacement Comparison (Adjusted)');
xlabel('Displacement (mm)');
ylabel('Force (N)');
legend_entries = {};

for i = 1:length(result_folders)
    folder_name = result_folders(i).name;
    folder_path = fullfile(result_folders(i).folder, folder_name);  % Correct the full path
    data_folder = fullfile(folder_path, 'outputdata');  % Now correctly points to the 'outputdata' directory

    % Specific RF2 and U2 filenames
    rf2_file_name = 'RF2_BC3.txt';
    u2_file_name = 'U2_BC3.txt';

    % Read data from files
    rf2_data = read_data(fullfile(data_folder, rf2_file_name));
    u2_data = read_data(fullfile(data_folder, u2_file_name));

    % Filter and adjust data
    [rf2_data_filtered, u2_data_adjusted] = filter_and_adjust_data(rf2_data, u2_data, 1.0e-10);

    % Save filtered and adjusted data
    save_data(fullfile(data_folder, [rf2_file_name(1:end-4) '_zero_forces_removed.txt']), rf2_data_filtered);
    save_data(fullfile(data_folder, [u2_file_name(1:end-4) '_displacement_corrected.txt']), u2_data_adjusted);

    % Determine the line style and color
    line_style = line_styles{mod(i-1, length(line_styles)) + 1};
    line_color = line_colors(i, :);

    % Plot the adjusted data
    plot(u2_data_adjusted, rf2_data_filtered, 'LineWidth', 1, 'LineStyle', line_style, 'Color', line_color);

    % Annotate the line with a number
    text(u2_data_adjusted(end), rf2_data_filtered(end), num2str(i), 'FontSize', 6, 'Color', line_color);

    % Add legend entry with escaped underscores
    legend_entries{end+1} = [num2str(i) ' ' strrep(strrep(folder_name, '_results', ''), '_', '\_')];
end

% Add the legend to the adjusted comparison plot
legend(legend_entries, 'Location', 'eastoutside');

% Save the adjusted comparison plot
savefig(fullfile(compare_folder, 'Force-Displacement_Comparison_displ_adjusted'));
saveas(gcf, fullfile(compare_folder, 'Force-Displacement_Comparison_displ_adjusted.png'));
saveas(gcf, fullfile(compare_folder, 'Force-Displacement_Comparison_displ_adjusted.svg'));

hold off;

% Plot and save original mIFSS-Displacement data
figure;
hold on;
title('mIFSS-Displacement Comparison');
xlabel('Displacement (mm)');
ylabel('apparent (mean) IFSS (N/mm^2)');
legend_entries = {};

for i = 1:length(result_folders)
    folder_name = result_folders(i).name;
    folder_path = fullfile(result_folders(i).folder, folder_name);  % Correct the full path
    data_folder = fullfile(folder_path, 'outputdata');  % Now correctly points to the 'outputdata' directory

    % Specific mIFSS and U2 filenames
    mIFSS_file_name = 'mIFSS_from_RF2_BC3.txt';
    u2_file_name = 'U2_BC3.txt';

    % Read data from files
    mIFSS_data = read_data(fullfile(data_folder, mIFSS_file_name));
    u2_data = read_data(fullfile(data_folder, u2_file_name));

    % Determine the line style and color
    line_style = line_styles{mod(i-1, length(line_styles)) + 1};
    line_color = line_colors(i, :);

    % Plot the original data
    plot(u2_data, mIFSS_data, 'LineWidth', 1, 'LineStyle', line_style, 'Color', line_color);

    % Annotate the line with a number
    text(u2_data(end), mIFSS_data(end), num2str(i), 'FontSize', 6, 'Color', line_color);

    % Add legend entry with escaped underscores
    legend_entries{end+1} = [num2str(i) ' ' strrep(strrep(folder_name, '_results', ''), '_', '\_')];
end

% Add the legend to the original mIFSS-Displacement comparison plot
legend(legend_entries, 'Location', 'eastoutside');

% Save the original mIFSS-Displacement comparison plot
savefig(fullfile(compare_folder, 'mIFSS-Displacement_Comparison'));
saveas(gcf, fullfile(compare_folder, 'mIFSS-Displacement_Comparison.png'));
saveas(gcf, fullfile(compare_folder, 'mIFSS-Displacement_Comparison.svg'));

hold off;

% Plot and save adjusted mIFSS-Displacement data
figure;
hold on;
title('mIFSS-Displacement Comparison (Adjusted)');
xlabel('Displacement (mm)');
ylabel('apparent (mean) IFSS (N/mm^2)');
legend_entries = {};

for i = 1:length(result_folders)
    folder_name = result_folders(i).name;
    folder_path = fullfile(result_folders(i).folder, folder_name);  % Correct the full path
    data_folder = fullfile(folder_path, 'outputdata');  % Now correctly points to the 'outputdata' directory

    % Specific mIFSS and U2 filenames
    mIFSS_file_name = 'mIFSS_from_RF2_BC3.txt';
    u2_file_name = 'U2_BC3.txt';

    % Read data from files
    mIFSS_data = read_data(fullfile(data_folder, mIFSS_file_name));
    u2_data = read_data(fullfile(data_folder, u2_file_name));

    % Filter and adjust data
    [mIFSS_data_filtered, u2_data_adjusted] = filter_and_adjust_data(mIFSS_data, u2_data, 1.0e-10);

    % Save filtered and adjusted data
    save_data(fullfile(data_folder, [mIFSS_file_name(1:end-4) '_zero_stresses_removed.txt']), mIFSS_data_filtered);
    %save_data(fullfile(data_folder, [u2_file_name(1:end-4) '_displacement_corrected.txt']), u2_data_adjusted);

    % Determine the line style and color
    line_style = line_styles{mod(i-1, length(line_styles)) + 1};
    line_color = line_colors(i, :);

    % Plot the adjusted data
    plot(u2_data_adjusted, mIFSS_data_filtered, 'LineWidth', 1, 'LineStyle', line_style, 'Color', line_color);

    % Annotate the line with a number
    text(u2_data_adjusted(end), mIFSS_data_filtered(end), num2str(i), 'FontSize', 6, 'Color', line_color);

    % Add legend entry with escaped underscores
    legend_entries{end+1} = [num2str(i) ' ' strrep(strrep(folder_name, '_results', ''), '_', '\_')];
end

% Add the legend to the adjusted mIFSS-Displacement comparison plot
legend(legend_entries, 'Location', 'eastoutside');

% Save the adjusted mIFSS-Displacement comparison plot
savefig(fullfile(compare_folder, 'mIFSS-Displacement_Comparison_displ_adjusted'));
saveas(gcf, fullfile(compare_folder, 'mIFSS-Displacement_Comparison_displ_adjusted.png'));
saveas(gcf, fullfile(compare_folder, 'mIFSS-Displacement_Comparison_displ_adjusted.svg'));

hold off;



% Plot and save adjusted mIFSS-Displacement data with maximum force points
figure;
hold on;
title('mIFSS_{max} (with displacement adjusted data)');
xlabel('Displacement (mm)');
ylabel('mIFSS_{max} (N/mm²)');
legend_entries = {};

% Set axes limits to start at 0
xlim([0 inf]);
ylim([0 inf]);

% Define a list of point styles
point_styles = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};

for i = 1:length(result_folders)
    folder_name = result_folders(i).name;
    folder_path = fullfile(result_folders(i).folder, folder_name);  % Correct the full path
    data_folder = fullfile(folder_path, 'outputdata');  % Now correctly points to the 'outputdata' directory

    % Specific adjusted mIFSS and U2 filenames
    mIFSS_file_name = 'mIFSS_from_RF2_BC3_zero_stresses_removed.txt';
    u2_file_name = 'U2_BC3_displacement_corrected.txt';

    % Read data from files
    mIFSS_data = read_data(fullfile(data_folder, mIFSS_file_name));
    u2_data = read_data(fullfile(data_folder, u2_file_name));

    % Find the index of the maximum force
    [max_force, max_idx] = max(mIFSS_data);

    % Get the corresponding displacement
    max_displacement = u2_data(max_idx);

    % Determine the point style and color
    point_style = point_styles{mod(i-1, length(point_styles)) + 1};
    point_color = line_colors(i, :);

    % Plot the maximum force point
    plot(max_displacement, max_force, 'Marker', point_style, 'LineStyle', 'none', 'Color', point_color, 'MarkerSize', 8);

    % Annotate the point with a number
    text(max_displacement, max_force, num2str(i), 'FontSize', 8, 'Color', point_color);

    % Add legend entry with escaped underscores
    legend_entries{end+1} = [num2str(i) ' ' strrep(strrep(folder_name, '_results', ''), '_', '\_')];
end

% Add the legend to the maximum force points plot
legend(legend_entries, 'Location', 'eastoutside');

% Save the maximum force points plot
savefig(fullfile(compare_folder, 'mIFSS_max_Comparison'));
saveas(gcf, fullfile(compare_folder, 'mIFSS_max_Comparison.png'));
saveas(gcf, fullfile(compare_folder, 'mIFSS_max_Comparison.svg'));

hold off;




%%%% Function to read data
function data = read_data(file_name)
    file = fopen(file_name, 'r');
    if file == -1
        error('Could not open file: %s', file_name);
    end
    data = textscan(file, '%s');
    fclose(file);
    data = str2double(data{1}(1:1:end));
end

%%%% Function to save data
function save_data(file_name, data)
    file = fopen(file_name, 'w');
    fprintf(file, '%.15g\n', data);
    fclose(file);
end

%%%% Function to filter and adjust data
function [rf2_filtered, u2_adjusted] = filter_and_adjust_data(rf2, u2, threshold)
    mask = abs(rf2) >= threshold;
    first_non_zero_index = find(mask, 1);
    
    % Adjust index if it is 1 to avoid invalid access
    if isempty(first_non_zero_index) || first_non_zero_index == 1
        first_non_zero_index = 2;  % Start from the second element
    end
    
    rf2_filtered = rf2(first_non_zero_index:end);
    u2_adjusted = u2(first_non_zero_index:end) - u2(first_non_zero_index-1);
    
    % Prepend 0 only once to the filtered data
    rf2_filtered = [0; rf2_filtered];
    u2_adjusted = [0; u2_adjusted];
end
