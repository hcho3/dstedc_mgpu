function test_suite(NGPU)

for i=[1024 2048 4096 8192]
    fprintf(1, 'i = %d\n', i);
    tester(NGPU, i);
    fprintf(1, '\n');
end
