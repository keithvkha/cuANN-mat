function x = discPulse(n, pulseN, delay)
% discrete Rect Pulse p_pulseN(n - delay);
% where n = input vector

if ~exist('delay', 'var')
    delay = 0;
end

% 1st implementation
%last = length(n);
%x = [zeros(size(n(1):delay-1), ones(size(delay:pulseN-1)), zeros(size(pulseN:n(last))))];

% 2nd implementation
% p(n-delay) = u(n - delay) - u(n - delay - pulseN);
x = discStep(n,delay) - discStep(n, delay + pulseN);

end