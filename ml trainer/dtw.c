#define MIN(x,y,z) ( x < y ? x : (y < z ? y : z) )
#define INDEX(x,y) ( x + y*n )
#define D(x,y) ( abs(x - y) )

double DTW(double *a1, double *a2, int n) {
    int i, j;
    double *cost, min;

    cost = malloc(sizeof(double)*n*n);

    cost[0] = D(a1[0], a2[0]);
    for (i = 0; i < n*n; i += n) {
        cost[i] = cost[i-n] + D(a1[i], a2[0]);
    }
    for (j = 0; j < n; j++) {
        cost[j] = cost[j-1] + D(a1[0], a2[j]);
    }

    for (i = 1; i < n; i++) {
        for (j = 1; j < n; j++) {
            min = MIN(cost[INDEX(i-1,j-1)],
                      cost[INDEX(i-1,j)],
                      cost[INDEX(i,j-1)]);
            cost[INDEX(i,j)] = min + D(a1[i], a2[j]);
        }
    }

    return cost[INDEX(n,n)];
}