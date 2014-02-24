addpath('../matio');
if exist('dgemm', 'dir') ~= 7
    error('dgemm/ directory must exist.');
end
if length(dir('dgemm')) > 2
    fprintf(1, 'Warning: dgemm/ directory is not empty. It may contain ');
    fprintf(1, 'some test matrices already.\n');
    prompt = 'Are you sure you want to create new test matrices? Y/N [Y]: ';
    str = input(prompt, 's');
    if isempty(str)
        str = 'Y';
    end
    if strncmpi(str, 'N', 1)
        fprintf(1, 'aborting...\n');
        return;
    end
end
for matsiz=[512 1024 2048 4096 8192]
    matsiz
    write_bin(sprintf('dgemm/A_%d.bin', matsiz), randn(matsiz));
    write_bin(sprintf('dgemm/B_%d.bin', matsiz), randn(matsiz));
end
