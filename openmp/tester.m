
v = sort(read_bin('v_16384.bin'));
for ngrp=[1 2 3 4 6 8 12 24]
    system(sprintf('./tester %d %d D_16384.bin E_16384.bin Dout.bin Q.bin',...
        ngrp, 24));
    D = read_bin('Dout.bin');

    max(abs(v-D))
end
