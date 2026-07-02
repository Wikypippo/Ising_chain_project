#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

double* calcola_ground_state_matrix_free(int N, double J, double h, double* energia_out) {
    long int dim = 1L << N;
    double* psi = (double*)malloc(dim * sizeof(double));
    double* temp = (double*)malloc(dim * sizeof(double));
    
    // Inizializzazione uniforme
    double norm_iniziale = sqrt(dim);
    #pragma omp parallel for
    for (long int i = 0; i < dim; i++) psi[i] = 1.0 / norm_iniziale;

    //Parametri per la convergenza
    double dt = 0.02;
    int max_iter = 15000;

    for (int iter = 0; iter < max_iter; iter++) {
        // EVOLUZIONE MATRIX-FREE 
        #pragma omp parallel for schedule(static)
        for (long int state = 0; state < dim; state++) {
            //Calcolo valori sulla diagonale
            double diag_val = 0.0;
            for (int k = 0; k < N; k++) {
                int bit_k = (state >> k) & 1;
                int bit_next = (state >> ((k + 1) % N)) & 1;
                diag_val += (bit_k == bit_next) ? -J : J;
            }
            //Calcolo valori fuori diagonale
            double sum = diag_val * psi[state];
            for (int k = 0; k < N; k++) {
                long int flipped = state ^ (1L << k);
                sum += (-h) * psi[flipped];
            }
            temp[state] = sum;
        }

        // Aggiornamento e Normalizzazione
        double norm2 = 0.0;
        #pragma omp parallel for reduction(+:norm2)
        for (long int i = 0; i < dim; i++) {
            psi[i] -= dt * temp[i];
            norm2 += psi[i] * psi[i];
        }
        double norm = sqrt(norm2);
        #pragma omp parallel for if(dim > 500)
        for (long int i = 0; i < dim; i++) psi[i] /= norm;
    }

    // Calcolo energia finale
    double energia = 0.0;
    #pragma omp parallel for reduction(+:energia) schedule(static) 
    for (long int state = 0; state < dim; state++) {
        double diag_val = 0.0;
        for (int k = 0; k < N; k++) {
            int bit_k = (state >> k) & 1;
            int bit_next = (state >> ((k + 1) % N)) & 1;
            diag_val += (bit_k == bit_next) ? -J : J;
        }
        double sum = diag_val * psi[state];
        for (int k = 0; k < N; k++) sum += (-h) * psi[state ^ (1L << k)];
        energia += psi[state] * sum;
    }
    *energia_out = energia;
    free(temp);
    return psi;
}

double calcola_Mz2(int N, double* GS) {
    long int dim = 1L << N;
    double m2 = 0.0;
    for (long int state = 0; state < dim; state++) {
        double Mz_stato = 0.0;
        for (int i = 0; i < N; i++) {
            int bit = (state >> i) & 1;
            Mz_stato += (bit == 0) ? 1.0 : -1.0;
        }
        Mz_stato /= N; 
        m2 += (GS[state] * GS[state]) * (Mz_stato * Mz_stato); 
    }
    return m2;
}

int main(int argc, char** argv) {
    int rank = 0, size = 1;
    double start_time;
    
    // Inizzializzazione
    #ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif

    int N_spin = (argc > 1) ? atoi(argv[1]) : 10;
    double J = 1.0;
    int num_steps = 21;

    int base_steps = num_steps / size;
    int remainder = num_steps % size;
    int local_steps = base_steps + (rank < remainder ? 1 : 0);
    int start_step = rank * (num_steps / size) + (rank < remainder ? rank : remainder);

    double* local_h = malloc(local_steps * sizeof(double));
    double* local_E = malloc(local_steps * sizeof(double));
    double* local_Mz = malloc(local_steps * sizeof(double));

    // Vettori globali per riordinare l'output
    double *global_h = NULL, *global_E = NULL, *global_Mz = NULL;
    int *recvcounts = NULL, *displs = NULL;

    // Allocazione della memoria da parte del main 
    if (rank == 0) {
        global_h = malloc(num_steps * sizeof(double));
        global_E = malloc(num_steps * sizeof(double));
        global_Mz = malloc(num_steps * sizeof(double));
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        int offset = 0;
        for (int p = 0; p < size; p++) {
            recvcounts[p] = base_steps + (p < remainder ? 1 : 0);
            displs[p] = offset;
            offset += recvcounts[p];
        }
        printf("Avviata simulazione per N_spin = %d (Processi MPI: %d)\n", N_spin, size);
    }

    #ifdef USE_MPI
    start_time = MPI_Wtime();
    #else
    start_time = omp_get_wtime();
    #endif

    // Calcolo
    for (int i = 0; i < local_steps; i++) {
        double h_corrente = (start_step + i) * 0.05;
        local_h[i] = h_corrente;
        double energia = 0.0;
        double* GS = calcola_ground_state_matrix_free(N_spin, J, h_corrente, &energia);
        local_E[i] = energia;
        local_Mz[i] = calcola_Mz2(N_spin, GS);
        free(GS);
    }

    // Raccoglimento dei risultati
    #ifndef USE_MPI
    for(int i = 0; i < local_steps; i++) {
        global_h[i] = local_h[i];
        global_E[i] = local_E[i];
        global_Mz[i] = local_Mz[i];
    }
    #endif

    #ifdef USE_MPI
    // Il Master raccoglie i dati da tutti e li riordina automaticamente
    MPI_Gatherv(local_h, local_steps, MPI_DOUBLE, global_h, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_E, local_steps, MPI_DOUBLE, global_E, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_Mz, local_steps, MPI_DOUBLE, global_Mz, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start_time;
    #else
    double elapsed = omp_get_wtime() - start_time;
    #endif

    // Solo il Master stampa i risultati riordinati
    if (rank == 0) {
        printf("Tempo totale di esecuzione: %f secondi\n", elapsed);
        printf("h,Energia,Mz2\n");
        for (int i = 0; i < num_steps; i++) {
            printf("%.2f,%f,%f\n", global_h[i], global_E[i], global_Mz[i]);
        }
        free(global_h); free(global_E); free(global_Mz);
        free(recvcounts); free(displs);
    }

    // Deallocazione della memoria
    free(local_h); free(local_E); free(local_Mz);
    #ifdef USE_MPI
    MPI_Finalize();
    #endif
    return 0;
}