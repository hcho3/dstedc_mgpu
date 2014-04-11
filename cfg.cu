#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dstedc.h"

static int try_read(FILE *cfg, const char *format, double *param)
{
    char buf[BUFSIZ] = {0};
    int flag = 0;

    while (fgets(buf, BUFSIZ, cfg)) {
        if (sscanf(buf, format, param) == 1) {
            flag = 1;
            break;
        }
    }

    return flag;
}

static void invalid(const char *msg)
// quit program due to invalid config file
{
    printf("Error: params.cfg is not a valid configuration file.\n");
    printf("       %s\n", msg);
    exit(1);
}

cfg_ent load_cfg(const char *filename)
{
    FILE *cfg = fopen(filename, "r");
    cfg_ent c;
    int Gvalid[6] = {0}, Cvalid[6] = {0};

    if (!cfg) {
        printf("Error: params.cfg has not been generated.\n");
        printf("Please run ./profile.\n");
        perror("fopen");
        exit(1);
    }

    Gvalid[0] = try_read(cfg, "Gparam[0] = %lf", &c.Gparam[0]);
    if (!Gvalid[0]) invalid("Gparam[0] was not found.\n");
    Gvalid[1] = try_read(cfg, "Gparam[1] = %lf", &c.Gparam[1]);
    if (!Gvalid[1]) invalid("Gparam[1] was not found.\n");
    Gvalid[2] = try_read(cfg, "Gparam[2] = %lf", &c.Gparam[2]);
    if (!Gvalid[2]) invalid("Gparam[2] was not found.\n");

    Cvalid[0] = try_read(cfg, "Cparam[0] = %lf", &c.Cparam[0]);
    if (!Cvalid[0]) invalid("Cparam[0] was not found.\n");
    Cvalid[1] = try_read(cfg, "Cparam[1] = %lf", &c.Cparam[1]);
    if (!Cvalid[1]) invalid("Cparam[1] was not found.\n");
    Cvalid[2] = try_read(cfg, "Cparam[2] = %lf", &c.Cparam[2]);
    if (!Cvalid[2]) invalid("Cparam[2] was not found.\n");

    Gvalid[3] = try_read(cfg, "Gparam[3] = %lf", &c.Gparam[3]);
    if (!Gvalid[3]) invalid("Gparam[3] was not found.\n");
    Gvalid[4] = try_read(cfg, "Gparam[4] = %lf", &c.Gparam[4]);
    if (!Gvalid[4]) invalid("Gparam[4] was not found.\n");
    Gvalid[5] = try_read(cfg, "Gparam[5] = %lf", &c.Gparam[5]);
    if (!Gvalid[5]) invalid("Gparam[5] was not found.\n");

    Cvalid[3] = try_read(cfg, "Cparam[3] = %lf", &c.Cparam[3]);
    if (!Cvalid[3]) invalid("Cparam[3] was not found.\n");
    Cvalid[4] = try_read(cfg, "Cparam[4] = %lf", &c.Cparam[4]);
    if (!Cvalid[4]) invalid("Cparam[4] was not found.\n");
    Cvalid[5] = try_read(cfg, "Cparam[5] = %lf", &c.Cparam[5]);
    if (!Cvalid[5]) invalid("Cparam[5] was not found.\n");

    fclose(cfg);

    return c;
}

int compute_dlaed1_partition(const cfg_ent cfg, const int NGPU,
    const int NCPU, const int N, const int subpbs)
{
    double c, g;
    double ratio;
    int gpu_portion;

    c = pow((double)N, cfg.Cparam[0]) * pow((double)NCPU, cfg.Cparam[1])
           * exp2(cfg.Cparam[2]);
    g = pow((double)N, cfg.Gparam[0]) * pow((double)NGPU, cfg.Gparam[1])
           * exp2(cfg.Gparam[2]);

    ratio = c / g;
    gpu_portion = (int)ceil(subpbs*ratio/(ratio+1));

    /* adjust work portion to GPUs a little bit to make it a multiple of NGPU.
     * Otherwise, one or more GPU device will be left idle. */
    //if (gpu_portion >= 1 && gpu_portion < NGPU)
    //    gpu_portion = (subpbs >= NGPU) ? NGPU : subpbs;
    //else if (gpu_portion > NGPU)
    //    gpu_portion -= gpu_portion % NGPU;

    return gpu_portion;
}

int compute_dlaed4_partition(const cfg_ent cfg, const int NGPU,
    const int NCPU, const int K)
{
    double c, g;
    double ratio;
    int gpu_portion;

    c = pow((double)K, cfg.Cparam[0]) * pow((double)NCPU, cfg.Cparam[1])
           * exp2(cfg.Cparam[2]);
    g = pow((double)K, cfg.Gparam[0]) * pow((double)NGPU, cfg.Gparam[1])
           * exp2(cfg.Gparam[2]);

    ratio = c / g;
    gpu_portion = (int)ceil(K*ratio/(ratio+1));

    return gpu_portion;
}
