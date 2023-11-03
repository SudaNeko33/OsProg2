/*
PLEASE WRITE DOWN NAME AND UID BELOW BEFORE SUBMISSION
* NAME:
* UID :

Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o template template.c utilities.c -O2 -pthread -lm

Then Run with:
$ ./parallel
*/

#define _GNU_SOURCE // keep this line
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 * 
 * Matrix-vector multiplication, used in QKV Mapping and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is 
 * independent of each other, so we can use parallel computing for acceleration.
 * 
 * Please use <pthread.h> and your favorite control method,
 * semaphore (please #include <semaphore.h>) / mutex lock + conditional variable
 * 
 * A sequential version is provided below, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE

// additional header file
#include <pthread.h>
#include <semaphore.h>


// global variables
struct rusage main_usage;        // get usage for main thread
int thr_count = 4; // number of threads to use
pthread_t threads[thr_count];
int thread_ids[thr_count];

pthread_mutex_t mutex;
pthread_cond_t cond;

int chunk_size = 0;

sem_t semaphore;

int init_mat_vec_mul(int thr_count) {
    

    for (int i = 0; i < thr_count; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thr_func, &thread_ids[i]);
        pthread_detach(threads[i]);
    }

    // Let threads fall asleep immediately
    sleep(1);

    return 0;
}


void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    // additional variables for threading

    chunk_size = row / thr_count; // number of rows to assign to each thread
    int remainder = row % thr_count; // remainder rows to be assigned to the last thread

    float* out_threads[thr_count];
    float* vec_threads[thr_count];
    float* mat_threads[thr_count];
    int start_row = 0;

    // assign parameters to threads
    for (int i = 0; i < thr_count; i++) {
        thread_ids[i] = i;
        out_threads[i] = out + start_row * col;
        vec_threads[i] = vec;
        mat_threads[i] = mat + start_row * col;
        if (i == thr_count - 1) {
            chunk_size += remainder; // add remainder rows to the last thread
        }
        start_row += chunk_size;
    }

    // wait for threads to complete
    for (int i = 0; i < thr_count; i++) {
        pthread_join(threads[i], NULL);
    }
}





int close_mat_vec_mul() {
    // wake up threads to collect system usage and terminate
    pthread_exit(NULL);

    // wait for all threads to exit and collect system usage
    for (int i = 0; i < thr_count; i++) {
        pthread_join(threads[i], NULL);
    }

    // clear all other resources related with multi-threading
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    sem_destroy(&semaphore);

    return 0;
}

void *thr_func(void *arg) {
    int thread_id = *(int*)arg;

    // perform matrix-vector multiplication for assigned rows
    for (int i = thread_id * chunk_size; i < (thread_id + 1) * chunk_size; i++) {
        float sum = 0;
        for (int j = 0; j < col; j++) {
            sum += mat_threads[thread_id][j] * vec_threads[thread_id][j];
        }
        out_threads[thread_id][i] = sum;
    }

    // collect system usage of the thread
    struct rusage usage;
    getrusage(RUSAGE_THREAD, &usage);
    printf("Thread %d: User CPU time used = %ld.%06ld sec, System CPU time used = %ld.%06ld sec\n",
           thread_id, usage.ru_utime.tv_sec, usage.ru_utime.tv_usec, usage.ru_stime.tv_sec, usage.ru_stime.tv_usec);

    pthread_exit(NULL);
}


// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}