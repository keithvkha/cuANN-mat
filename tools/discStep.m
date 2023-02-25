function x = discStep(n, delay)
% discrete Step/Heaviside function u(n - delay)
% where n = input vector

if ~exist('delay', 'var')
    delay = 0;
end

x = [zeros(1,delay), ones(1,length(n)-delay)]';


% Convert to column vector
% nSize = size(n);
% if(nSize(1) == 1)
%     x = x';
% end


end