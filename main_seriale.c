#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

double** costruisci_hamiltoniana_ising(int N, double J, double h) {
    long int dim = 1L << N; // Dimensione 2^N
    
    // 1. Allocazione della matrice e inizializzazione a zero
    double** H = (double**)malloc(dim * sizeof(double*));
    if (H == NULL) {
        printf("Errore di allocazione memoria per le righe.\n");
        return NULL;
    }
    for (int i = 0; i < dim; i++) {
        H[i] = (double*)calloc(dim, sizeof(double)); 
        if (H[i] == NULL) {
            printf("Errore di allocazione memoria per le colonne.\n");
            return NULL;
        }
    }  

    // 2. Popolamento dell'Hamiltoniana
    for (long int state = 0; state < dim; state++) {
        double diag_val = 0.0;
        // Il ciclo arriva fino a N per creare anche il termine di raccordo tra N-1 e 0 
        for (int i = 0; i < N; i++) {
            int bit_i = (state >> i) & 1;            
            int bit_j = (state >> ((i + 1) % N)) & 1;
            
            // Se i bit sono uguali (entrambi 0 o entrambi 1), il prodotto è +1. Altrimenti -1.
            diag_val += (bit_i == bit_j) ? -J : J;
        }
        H[state][state] = diag_val;

        // Termine di campo trasverso X 
        for (int i = 0; i < N; i++) {
            long int flipped_state = state ^ (1L << i);
            H[state][flipped_state] = -h;
        }
    }

    return H;
}

double* calcola_stato_ground(double** H, int N, double* energia_out) {
    long int dim = 1L << N;
    
    // Allocazione dell'autovettore che verrà restituito
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
        for (long int i = 0; i < dim; i++) {
            double somma_locale = 0.0;
            for (long int j = 0; j < dim; j++) {
                somma_locale += H[i][j] * psi[j];
            }
            temp[i] = somma_locale;
        }

        // psi = psi - dt * temp
        double norm2 = 0.0;
        for (long int i = 0; i < dim; i++) {
            psi[i] = psi[i] - dt * temp[i];
            norm2 += psi[i] * psi[i];
        }

        // Normalizzazione
        double norm = sqrt(norm2);
        for (long int i = 0; i < dim; i++) {
            psi[i] /= norm;
        }
    }

    // Calcolo dell'energia finale E = <psi | H | psi>
    double energia = 0.0;
    for (long int i = 0; i < dim; i++) {
        double somma_locale = 0.0;
        for (long int j = 0; j < dim; j++) {
            somma_locale += H[i][j] * psi[j];
        }
        energia += psi[i] * somma_locale;
    }

    // Esporto l'energia tramite il puntatore
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

int main(int argc, char*argv[]){
    if (argc < 2) {
        printf("Uso: %s <numero_di_spin_N>\n", argv[0]);
        return 1;
    }

    // Parametri dell'Hamiltoniana
    const int N = atoi(argv[1]);
    double J = 1.0;
    long int dim = (1<<N);
    double start_time;


    printf("h,Energia,Mz2\n");

    start_time = omp_get_wtime();  
    // Variazione di h da 0.0 a 1.0 p
    for (double h = 0.0; h <= 1.01; h += 0.05) {
        
        // Costruzione di H
        double** H = costruisci_hamiltoniana_ising(N, J, h);
        if (H == NULL) {
            printf("Errore di memoria.\n");
            return 1;
        }

        // calcolo stato ground ed energia
        double energia_minima = 0.0;
        double* autovettore = calcola_stato_ground(H, N, &energia_minima);

        // Calcolo Magnetizzazione
        double Mz_2 = calcola_Mz2(N, autovettore);

        printf("%.2f,%f,%f\n", h, energia_minima, Mz_2);

        for (long int i = 0; i < dim; i++) {
            free(H[i]);
        }
        free(H);
        free(autovettore);
    }

    double elapsed = omp_get_wtime() - start_time;
    printf("Tempo totale di esecuzione: %f secondi\n",elapsed);

    return 0;
}