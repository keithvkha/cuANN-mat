function index = extractIndex(key, array)
    % This function only finds/extracts the index for an element that
    % exists in the array
    % i.e. It assumes the "key" exists in the "array"
    last = length(array);
    if (last == 1)      % meaning the array only has 1 element
        index = 1;
    else
        % Using the scaling equation
        index = round(1 + (key - array(1))./(array(last)-array(1)).*(last-1));

%         Convention for array of index: column vector
        if (size(index,1) == 1)      % if a row vector
            index = index';         % convert to column vector
        end

        % Convention for array of index: row vector
%         if (size(index,2) == 1)      % if a column vector
%             index = index';         % convert to row vector
%         end
    end
end