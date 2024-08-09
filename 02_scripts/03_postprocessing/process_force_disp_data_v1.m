close all

%this script takes the output variables RF & U of the Reference Point and
%plots them

%get from shell-script:
%jobname = 'Meniscus90_02rf_v1';
%inc=10;

%embedded_area = 0.014515;

%Test if jobname is correct
disp(['Jobname is: ', jobname])

folder = append(jobname,"_results");
global resultfiles
resultfiles = './' + folder + '/';

global datafolder
datafolder = resultfiles + 'outputdata/';

global lit_folder
lit_folder = "./_literature_data/";

%%% names
time_RF = "time_RF";

RF2 = 'RF2_BC3';
U2 = 'U2_BC3';

%%% read data 
steptime = read_data(time_RF);

data_RF2 = read_data(RF2);

data_U2 = read_data(U2);


%%% calculate stress / strain
%s22 = -data_RF2_RP1./A_0;

%e22 = U_diff./l_0;


hPlot = plot(data_U2, data_RF2, 'DisplayName', 'RF2 vs U2', 'LineWidth', 1);
hold on  

%plot lit data
%solely MAPP
%e11_lit = read_data_lit('MAPP_e11.txt');
%s11_lit = read_data_lit('MAPP_s11.txt');
%e11_real = e11_lit./100;

%hPlot = plot(e11_real, s11_lit, 'DisplayName', 's11\_LIT\_MAPP', 'LineWidth', 1.5);



%yline(0,'HandleVisibility','off') %do not show in legend
%xline(0,'HandleVisibility','off')

hold off

% rename, so the underscore for the legend is interpreted correctly
jobname_underscore = regexprep(jobname, '[\\\^\_]','\\$0');

title(jobname_underscore)
legend('Location','eastoutside')
xlabel('displacement in mm')
ylabel('force in N')
%xlabel('strain')
%ylabel('stress [MPa]')

savefig(resultfiles + 'Force-Disp')
saveas(gcf, resultfiles + 'Force-Disp'+'.png')
saveas(gcf, resultfiles + 'Force-Disp'+'.svg')
hold on

%%%%%%%%%%%%%%%%%% include increments as vertical lines %%%%%%%%%%%%%%%%%
inc_1 = inc+1;

disp_X = data_U2(inc_1:inc:end);
incs = (1:length(data_U2))';
incs_X = incs(inc:inc:end);
incs_X = num2str(incs_X);

for i = 1:length(disp_X)
    incnum = incs_X(i,:);
    incline = xline(disp_X(i),'--r',incnum,'HandleVisibility','off');
    incline.LabelHorizontalAlignment = 'center';
end

hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inc_str = num2str(inc);

savefig(resultfiles + 'Force-Disp_inc' + inc_str)
saveas(gcf, resultfiles + 'Force-Disp_inc' + inc_str + '.png')
saveas(gcf, resultfiles + 'Force-Disp_inc' + inc_str + '.svg')


%%%% ADD mIFSS with EMBEDDED AREA INFORMATION %%%%

% Read the embedded area from the file
embedded_area_file = append(jobname, "_inp-files/xx__embedded_area__xx.txt");
fileID = fopen(embedded_area_file, 'r');
embedded_area = fscanf(fileID, '%f');
fclose(fileID);

% Calculate mIFSS
mIFSS = data_RF2 / embedded_area;

% Save mIFSS data to a new text file
mIFSS_file = datafolder + 'mIFSS_from_RF2_BC3.txt';
fileID = fopen(mIFSS_file, 'w');
fprintf(fileID, '%.6f\n', mIFSS);
fclose(fileID);

% Plot mIFSS vs Displacement
figure;
plot(data_U2, mIFSS, 'DisplayName', 'mIFSS vs U2', 'LineWidth', 1);
hold on

% Rename for legend display
jobname_underscore = regexprep(jobname, '[\\\^\_]','\\$0');

title(jobname_underscore)
legend('Location','eastoutside')
xlabel('displacement in mm')
ylabel('apparent (mean) IFSS (N/mm^2)')

% Save the new plot
savefig(resultfiles + 'mIFSS-Disp')
saveas(gcf, resultfiles + 'mIFSS-Disp'+'.png')
saveas(gcf, resultfiles + 'mIFSS-Disp'+'.svg')
hold off


%%%% functions: (at the end of the script file or in seperate files...
function data_string = read_data(RF_name)
    global datafolder
    file_name_long = datafolder + RF_name + '.txt';    
    file = fopen(file_name_long,'r');
    data = textscan(file,'%s');
    fclose(file);
    data_string = str2double(data{1}(1:1:end));
end

%%%% functions: (at the end of the script file or in seperate files...
function data_string = read_data_lit(RF_name)
    %global lit_folder
    file_name_long = RF_name;    
    file = fopen(file_name_long,'r');
    data = textscan(file,'%s');
    fclose(file);
    data_string = str2double(data{1}(1:1:end));
end
