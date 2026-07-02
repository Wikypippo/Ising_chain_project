#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#ifdef USE_MPI
#include <mpi.h>
#endif


double** costruisci_hamiltoniana_ising(int N, double J, double h) {
    long int dim = 1L << N; // Dimensione 2^N
    
    // 1. Allocazione della matrice densa e inizializzazione a zero
    double** H = (double**)malloc(dim * sizeof(double*));
    if (H == NULL) {
        printf("Errore di allocazione memoria per le righe.\n");
        return NULL;
    }
    
    for (int i = 0; i < dim; i++) {
        H[i] = (double*)calloc(dim, sizeof(double)); 
        if(H[i] == NULL){
            printf("Errore di allocazione memoria per le colonne.\n");
            for(int j=0; j<i; j++)
                free(H[j]);

            free(H);

            return NULL;
        }
    }

    // 2. Popolamento dell'Hamiltoniana
    #pragma omp parallel for schedule(static)
    for (long int state = 0; state < dim; state++) {
        
        double diag_val = 0.0;
        for (int i = 0; i < N; i++) {
            int bit_i = (state >> i) & 1;
            int bit_j = (state >> ((i + 1) % N)) & 1;
            
            // Se i bit sono uguali (entrambi 0 o 1), il prodotto è +1. Altrimenti -1.
            diag_val += (bit_i == bit_j) ? -J : J;
        }
        H[state][state] = diag_val;

        // Termine di campo trasverso X (Fuori Diagonale)
        for (int i = 0; i < N; i++) {
            // L'operatore XOR flippa l'i-esimo bit
            long int flipped_state = state ^ (1L << i);
            H[state][flipped_state] = -h;
        }
    }

    return H;
}

double* calcola_stato_ground(double** H, int N, double* energia_out) {
    long int dim = 1L << N;
    
    // Allocazione dell'autovettore
    double* psi = (double*)malloc(dim * sizeof(double));
    double* temp = (double*)malloc(dim * sizeof(double));
    
    if (psi == NULL || temp == NULL) {
        printf("Errore: memoria insufficiente per l'autovettore.\n");
        return NULL;
    }
    
    // Inizializzazione in sovrapposizione uniforme
    double norm_iniziale = sqrt(dim);
    for (long int i = 0; i < dim; i++) {
        psi[i] = 1.0 / norm_iniziale;
    }

    // Parametri per la convergenza esatta dell'energia
    double dt = 0.02;       
    int max_iter = 15000;   

    for (int iter = 0; iter < max_iter; iter++) {
        // temp = H * psi
        #pragma omp parallel for schedule(static) 
        for (long int i = 0; i < dim; i++) {
            double somma_locale = 0.0;
            for (long int j = 0; j < dim; j++) {
                somma_locale += H[i][j] * psi[j];
            }
            temp[i] = somma_locale;
        }

        // psi = psi - dt * temp
        double norm2 = 0.0;
        #pragma omp parallel for reduction(+:norm2)
        for (long int i = 0; i < dim; i++) {
            psi[i] = psi[i] - dt * temp[i];
            norm2 += psi[i] * psi[i];
        }

        // Normalizzazione
        double norm = sqrt(norm2);
        #pragma omp parallel for
        for (long int i = 0; i < dim; i++) {
            psi[i] /= norm;
        }
    }

    // Calcolo dell'energia finale E = <psi | H | psi>
    double energia = 0.0;
    #pragma omp parallel for reduction(+:energia) schedule(static)
    for (long int i = 0; i < dim; i++) {
        double somma_locale = 0.0; 
        for (long int j = 0; j < dim; j++) {
            somma_locale += H[i][j] * psi[j];
        }
        energia += psi[i] * somma_locale;
    }

    // Esporta l'energia tramite il puntatore
    if (energia_out != NULL) {
        *energia_out = energia;
    }

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
        
        double probabilita = GS[state] * GS[state];
        
        m2 += probabilita * (Mz_stato * Mz_stato); 
    }
    return m2;
}

int main(int argc, char** argv) {
    int rank = 0, size = 1;
    double start_time;
    
    // Inizializza MPI
    #ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif

    // Lettura di N_spin 
    int N_spin = 10; // Default
    if (argc > 1) {
        N_spin = atoi(argv[1]);
    }
    
    double J = 1.0;
    int num_steps = 21;
    
    // Divisione del lavoro MPI
    int base_steps = num_steps / size;
    int remainder = num_steps % size;

    int local_steps = base_steps + (rank < remainder ? 1 : 0);

    int start_step;

    if(rank < remainder)
        start_step = rank * (base_steps + 1);
    else
        start_step = remainder * (base_steps + 1) + (rank - remainder) * base_steps;
    
    // Array locali per salvare i risultati
    double* local_h = (double*)malloc(local_steps * sizeof(double));
    double* local_E = (double*)malloc(local_steps * sizeof(double)); 
    double* local_Mz = (double*)malloc(local_steps * sizeof(double));

    // Array globali per la raccolta finale
    double* global_h = NULL;
    double* global_E = NULL; 
    double* global_Mz = NULL;
    
    int *recvcounts = NULL;
    int *displs = NULL;

    // Allocazione della memoria dal main
    if(rank == 0)
    {
        global_h  = malloc(num_steps*sizeof(double));
        global_E  = malloc(num_steps*sizeof(double));
        global_Mz = malloc(num_steps*sizeof(double));

        recvcounts = malloc(size*sizeof(int));
        displs     = malloc(size*sizeof(int));

        int offset = 0;

        for(int p=0; p<size; p++)
        {
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

    // CICLO MPI LOCALE: Ogni processo calcola la sua fetta di valori h
    for (int i = 0; i < local_steps; i++) {
        int step_globale = start_step + i;
        
        double h_corrente = step_globale * 0.05; 
        
        local_h[i] = h_corrente;
        
        
        // Costruisce la matrice
        double** H_corrente = costruisci_hamiltoniana_ising(N_spin, J, h_corrente);
        
        // Calcola il Ground State e l'energia
        double energia_GS = 0.0;
        double* GS = calcola_stato_ground(H_corrente, N_spin, &energia_GS);
        
        // Calcola l'osservabile e salva l'energia
        local_E[i] = energia_GS; 
        local_Mz[i] = calcola_Mz2(N_spin, GS); 
        
        // Pulizia della memoria
        long int dim = 1L << N_spin;
        for (long int j = 0; j < dim; j++) {
            free(H_corrente[j]);
        }
        free(H_corrente);
        free(GS);
    }

    // Raccoglimento dei risultati
    #ifndef USE_MPI
    for(int i=0;i<local_steps;i++)
    {
        global_h[i] = local_h[i];
        global_E[i] = local_E[i]; 
        global_Mz[i] = local_Mz[i];
    }
    #endif

    #ifdef USE_MPI
    MPI_Gatherv(local_h, local_steps, MPI_DOUBLE, global_h, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_E, local_steps, MPI_DOUBLE, global_E, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_Mz, local_steps, MPI_DOUBLE, global_Mz, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start_time;
    #else
    double elapsed = omp_get_wtime() - start_time;
    #endif

    // Stampa dei risultati da parte del master e deallocazione
    if (rank == 0) {
        printf("Tempo totale di esecuzione: %f secondi\n",elapsed);
        printf("h,Energia,Mz2\n"); 
        for (int i = 0; i < num_steps; i++) {
            printf("%.2f,%f,%f\n", global_h[i], global_E[i], global_Mz[i]); 
        }
        free(global_h);
        free(global_E);
        free(global_Mz);
        free(recvcounts);
        free(displs);
    }

    free(local_h);
    free(local_E);
    free(local_Mz);
    #ifdef USE_MPI
    MPI_Finalize();
    #endif
    
    return 0;
}