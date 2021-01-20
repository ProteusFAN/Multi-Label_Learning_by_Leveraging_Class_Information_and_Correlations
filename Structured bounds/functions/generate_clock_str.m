%% generates a string from the current clock
function [c] = generate_clock_str()

temp = clock;
temp = round(temp(:));
c = sprintf('%d_%d_%d_%d_%d_%d',temp(1),temp(2),temp(3),temp(4),temp(5),temp(6));

return