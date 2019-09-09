#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "math.h"
#include "mpi.h"
#include "time.h"
#define NARG    3
#define NTRAIN  5000
#define NTEST   1000
#define NSTEP   100000
#define LRATE   0.2
#define PNLTY   0.1
#define ERRLMT  1e-3
#define FTRAIN  0
#define FTEST   1
#define SIGMOID(X) (1.0 / (1.0 + exp(-(X))))

// MPI并行进程数不能超出本地CPU核组数，否则会卡死

void gen_tdata(double *xt, double *y, double *wt, unsigned char flag, \
               int myid, int numprocs);
void train(double *xt, double *y, double *wt, int myid, int numprocs);
void test(double *wt);
void output(double *wt);

int main(int argc, char* argv[]) {
    double xt[NTRAIN * NARG];
    double y[NTRAIN];
    double wt[NARG + 1] = {0};
    int myid, numprocs;
    clock_t timer;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    if (myid == 0) timer = clock();
    gen_tdata(xt, y, wt, FTRAIN, myid, numprocs);
    train(xt, y, wt, myid, numprocs);
    if (myid == 0) {
        timer = clock() - timer;
        printf("\nTraining Time:\n\t%lf s\n", (double)timer / CLOCKS_PER_SEC);
        gen_tdata(xt, y, wt, FTEST, myid, numprocs);
        output(wt);
    }
    MPI_Finalize();

    //scanf("\n");
    return 0;
}

void gen_tdata(double *xt, double *y, double *wt, unsigned char flag, \
               int myid, int numprocs) {
    double wr[NARG] = {1.0, 0.9, 1.5};  // 得分、篮板、助攻权重
    double standard = 32.0;             // 定义“超巨”
    time_t t;
    srand((unsigned)time(&t));
    if (!flag) {
        for (int i = myid; i < NTRAIN; i += numprocs) {
            y[i] = 0.0;
            for (int j = 0; j < NARG; j++) {
                xt[i * NARG + j] = (double)(rand() % 20);
                y[i] += wr[j] * xt[i * NARG + j];
            }
            y[i] = (y[i] > standard) ? 1.0 : 0.0;
        }
    }
    else {
        int correct = 0;
        for (int i = 0; i < NTEST; i++) {
            double y0   = 0.0;
            double y1   = 0.0;
            for (int j = 0; j < NARG; j++) {
                xt[j] = (double)(rand() % 20);
                y0 += wr[j] * xt[j];
                y1 += wt[j] * xt[j];
            }
            y0 = (y0 > standard) ? 1.0 : 0.0;
            y1 = SIGMOID(y1 + wt[NARG]);
            //y1 = (y1 + wt[NARG] > 0.0) ? 1.0 : 0.0;
            if (fabs(y1 - y0) < 0.5) correct++;
        }
        printf("\nAccuracy:\n\t");
        printf("%-6.2lf%%\n", (double)correct * 100 / NTEST);
    }
    return;
}

void train(double *xt, double *y, double *wt, int myid, int numprocs) {
    for (int n = 0; n < NSTEP; n++) {
        double dw[NARG + 1] = {0};
        for (int i = myid; i < NTRAIN; i += numprocs) {
            double a = wt[NARG];
            for (int j = 0; j < NARG; j++) {
                a += wt[j] * xt[i * NARG + j];
            }
            a = SIGMOID(a);
            double dz = a - y[i];
            for (int j = 0; j < NARG; j++) {
                dw[j] += dz * xt[i * NARG + j];
            }
            dw[NARG] += dz;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double dwtmp[NARG + 1];
        MPI_Reduce(&dw[0], &dwtmp[0], NARG + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myid == 0) {
            for (int j = 0; j < NARG + 1; j++) {
                dw[j]  = dwtmp[j] + PNLTY * wt[j];
                dw[j] /= NTRAIN;
                wt[j] -= dw[j] * LRATE;
            }
        }
        MPI_Bcast((void *)&wt[0], NARG + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast((void *)&dw[0], NARG + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!((n + 1) % 1000)) {
            double maxdw = 0.0;
            for (int j = 0; j < NARG + 1; j++) {
                if (fabs(dw[j]) > maxdw) maxdw = fabs(dw[j]);
            }
            if (maxdw < ERRLMT) {
                if (myid == 0) printf("Iterated %d Times with Derivatives < %6.2e\n", \
                               n + 1, maxdw);
                return;
            }
        }
    }
    if (myid == 0) printf("Iterated %d Times (MAX TIME STEPS)\n", NSTEP);
    return;
}

void output(double *wt) {
    printf("\nWeight List:\n\t");
    for (int j = 0; j < (NARG + 1); j++) {
        if (!(j % 5)) {
            if (j) printf("\n\t");
        }
        printf("%-8.2lf, ", wt[j]);
    }
    return;
}
