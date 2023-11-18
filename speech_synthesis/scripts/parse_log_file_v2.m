function file_content = parse_log_file_v2(full_file_path)
    % Function for parsing the log-files from a full hyperparameter
    % optimization run. 
    % @param full_file_path (str): full file path to the log file.
    
    hyperparam_start_token = "checkpoint";
    train_start_token = "started_training";
    test_start_token = "started_testing";
    
    separator = ' ';
    
    % ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    file_dir = dir(full_file_path);
    if(isempty(file_dir))
        error("Could not locate files at %s", full_file_path);
    end
    
    file_name = sprintf("%s\\%s", file_dir(1).folder, file_dir(1).name);
       
    % Read in line by line.
    fid = fopen(file_name);
    
    lines = [];
    line_index = 1;

    while(~feof(fid))
       lines{line_index} = fgetl(fid);
       line_index = line_index + 1;
    end
   
    num_lines = line_index - 1;
    fclose(fid);
       
    % Decode the lines.
    line_index = 1;      
    run_index = 1;
    state = 0; % {parse_input, parse_hp, parse_train_res, parse_test_res}
    
    while(line_index <= num_lines)
        
        current_line = lines{line_index};
        
        switch state
            case 0 % Find the input selection start token.
                if(contains(current_line, hyperparam_start_token))
                    state = 1;
                end
                               
            case 1 % Hyperparameter selection.
                if(~contains(current_line, train_start_token))
                    if(contains(current_line, "vocoder_parameter_list") || ...
                       contains(current_line, "spectra_list") || ...
                       contains(current_line, "transforms_list") || ...
                       contains(current_line, "is_sequence_normalized"))
                        split_str = split(lines{line_index}, separator);
                        file_content{run_index}.hp.(split_str{1}) = split_str{2};
                    else                        
                        split_str = split(lines{line_index}, separator);
                        file_content{run_index}.hp.(split_str{1}) = str2num(split_str{2});
                    end
                else
                    epoch = 1;
                    state = 2;
                end
                
            case 2 % test results.
                if(~contains(current_line, test_start_token))
%                     if(convert_line)
                    floats = regexp(current_line,'[+-]?\d+\.?\d*', 'match');
                    file_content{run_index}.val_error(epoch) = str2num(floats{2});
                    file_content{run_index}.train_error(epoch) = str2num(floats{3});
                    epoch = epoch + 1;
%                     end
%                     convert_line = ~convert_line;
                else
                    state = 3;
                end
                
            case 3 % train results.
                if(~contains(current_line, hyperparam_start_token))
                    floats = regexp(current_line,'[+-]?\d+\.?\d*', 'match');
                    file_content{run_index}.test_error = str2num(floats{1});
                else
                    run_index = run_index + 1;
                    state = 1; % Back to reading hyperparameters if another run follows.
                end
        end
        
        line_index = line_index + 1;
    end
           
end

