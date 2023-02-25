function x = discDelta(n, delay)
% discrete Dirac delta function delta(n - delay)
% where n = input vector

if ~exist('delay', 'var')
    delay = 0;
end

x = [zeros(1,delay), 1, zeros(1,length(n)-1-delay)]';


% Convert to column vector
% nSize = size(n);
% if(nSize(1) == 1)       % if n is a row vector
%     x = x';             % then convert x to column vector
% end


end