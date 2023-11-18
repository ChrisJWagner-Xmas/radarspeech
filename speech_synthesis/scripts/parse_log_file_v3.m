function file_content = parse_log_file_v3(full_file_path)
    % Function for parsing the log-files from a full hyperparameter
    % optimization run. 
    % @param full_file_path (str): full file path to the log file.
    
    io_and_transform_opts_token = "io_and_transform_opts";
    hyperparam_start_token = "hyperparameters";
    training_start_token = "started_training";
    test_start_token = "started_testing";
    
    file_content = struct();
    
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
    read_values = false;
    epoch = 1;
    
    while(line_index <= num_lines)
        
        current_line = lines{line_index};
        
        switch state                             
            case 0 % io_and_transform_opts selection.
                if(contains(current_line, io_and_transform_opts_token))
                    dict_str = strrep(current_line, io_and_transform_opts_token, "");
                    d = strrep(dict_str, char(39), char(34)); % replace ' with "
                    d = char(d);
                    d = strrep(d,"True","1");
                    d = strrep(d,"False","0");
                    dict = jsondecode(d);
                    file_content.io_and_transforms = dict;
                    state = 1;
                end
                
            case 1 % hyperparam_start section.
                if(contains(current_line, hyperparam_start_token))
                    dict_str = strrep(current_line, hyperparam_start_token, "");
                    d = strrep(dict_str, char(39), char(34)); % replace ' with "
                    d = char(d);
                    d = strrep(d,"True","1");
                    d = strrep(d,"False","0");
                    dict = jsondecode(d);
                    file_content.hyperparameters = dict;
                    state = 2;
                end
                
            case 2 % test results.
                if(contains(current_line, test_start_token))
                    read_values = false;
                end
                if(read_values)
                    floats = regexp(current_line,'[+-]?\d+\.?\d*', 'match');
                    file_content.val_error(epoch) = str2num(floats{2});
                    file_content.train_error(epoch) = str2num(floats{3});
                    epoch = epoch + 1;
                end  
                if(contains(current_line, training_start_token))
                    read_values = true;
                end
        end
        
        line_index = line_index + 1;
    end
           
end

