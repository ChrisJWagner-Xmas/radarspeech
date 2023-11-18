function file_content = parse_log_file(full_file_path)
    % Function for parsing the log-files from a full hyperparameter
    % optimization run. 
    % @param full_file_path (str): full file path to the log file.
    
    input_start_token = "input_options";
    hyperparam_start_token = "hyperparameters";
    train_start_token = "started_training";
    test_start_token = "started_testing";
    
    
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
%     convert_line = true; % Temporary fix for alternating output lines.
    line_index = 1;      
    run_index = 1;
    state = 0; % {parse_input, parse_hp, parse_train_res, parse_test_res}
    
    while(line_index <= num_lines)
        
        current_line = lines{line_index};
        
        switch state
            case 0 % Find the input selection start token.
                if(~isempty(findstr(current_line, input_start_token)))
                    state = 1;
                end
                
            case 1 % Parse input selection.
                if(isempty(findstr(current_line, hyperparam_start_token)))
                    split_str = split(lines{line_index}, ' ');
                    file_content{run_index}.inputs.(split_str{1}) = split_str{2};
                else
                    state = 2;
                end
                
            case 2 % Hyperparameter selection.
                if(isempty(findstr(current_line, train_start_token)))
                    split_str = split(lines{line_index}, ' ');
                    file_content{run_index}.hp.(split_str{1}) = str2num(split_str{2});
                else
                    epoch = 1;
                    state = 3;
                end
                
            case 3 % test results.
                if(isempty(findstr(current_line, test_start_token)))
%                     if(convert_line)
                    floats = regexp(current_line,'[+-]?\d+\.?\d*', 'match');
                    file_content{run_index}.val_error(epoch) = str2num(floats{2});
                    file_content{run_index}.train_error(epoch) = str2num(floats{3});
                    epoch = epoch + 1;
%                     end
%                     convert_line = ~convert_line;
                else
                    state = 4;
                end
                
            case 4 % train results.
                if(isempty(findstr(current_line, hyperparam_start_token)))
                    floats = regexp(current_line,'[+-]?\d+\.?\d*', 'match');
                    file_content{run_index}.test_error = str2num(floats{1});
                else
                    run_index = run_index + 1;
                    state = 2; % Back to reading hyperparameters if another run follows.
                end
        end
        
        line_index = line_index + 1;
    end
           
end

