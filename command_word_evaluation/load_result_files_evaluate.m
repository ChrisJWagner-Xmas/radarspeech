clc;clear
addpath("C:\Programming\GitLab\rs_command_word_evaluation");
path_to_files = "C:\Programming\GitLab\rs_command_word_evaluation\results" ...
    + "\inter_session\no_cv_tested_code_train_val_set" ...
    + "\S002_split1_IV(2)";
train_metrics = {'batch_losses', 'validation_metric', 'train_metric', 'validation_accuracy'};
test_metrics = {'batch_losses', 'train_metric', 'test_accuracy', 'test_targets', 'test_predictions'};

[train_results, hp_list] = load_result_files(path_to_files, train_metrics, 'train');
test_results = load_result_files(path_to_files, test_metrics, 'test');

%% Plot results.
index = 1;
plot(train_results{index}.train_metric);
hold on
plot(train_results{index}.validation_metric);
hold off

%% Bundle results into table/array.
results_array = zeros(numel(test_results), numel(hp_list)+2);

for hp_index = 1:numel(hp_list)
    for result_index = 1:numel(test_results)
        results_array(result_index, hp_index) = ...
                            test_results{result_index}.(hp_list{hp_index});
        results_array(result_index, hp_index+1) = train_results{result_index}.validation_accuracy;
        results_array(result_index, hp_index+2) = test_results{result_index}.test_accuracy;
    end
end

%% Plot hyperparameters.
x_axis_index = 3;
y_axis_index = 8;
z_axis_index = size(results_array,2)-1;

plot3(results_array(:,x_axis_index), ...
    log10(results_array(:,y_axis_index)), ...
    results_array(:, z_axis_index), 'x');
grid on
hold on
plot3(results_array(:,x_axis_index), ...
    log10(results_array(:,y_axis_index)), ...
    results_array(:, z_axis_index+1), 'x');
xlabel(hp_list{x_axis_index}, 'interpreter','none')
ylabel(hp_list{y_axis_index}, 'interpreter','none')
legend('validation accuracies', 'test accuracies')

%% Plot training metrics.
index = 1;
plot(train_results{index}.validation_metric)
hold on
plot(train_results{index}.train_metric)
hold off

%% Display the confusion matrix.
% Load the information.
corpusFullFilePath = "C:\Users\chris\Documents\Institut\corpora\command_word_recognition";
addpath('C:\Programming\MATLAB\radarspeech\DtwErkenner_26-04-2021');

% Load the corpus information.
fid = fopen(sprintf("%s\\%s", corpusFullFilePath, "corpusInformation.json")); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
corpusInformation = jsondecode(str); 

% Build the confusion matrix.
index = 18;
test_predictions_numeric = test_results{index}.test_predictions+1; % +1: for matlab indexing
test_targets_numeric = test_results{index}.test_targets+1; % +1: for matlab indexing
classLabels = sort(corpusInformation.Classes);

confMatrix = confusionmat(test_targets_numeric, test_predictions_numeric);
displayConfusionMatrix(confMatrix, classLabels)

