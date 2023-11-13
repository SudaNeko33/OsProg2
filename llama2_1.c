/*
PLEASE WRITE DOWN NAME AND UID BELOW BEFORE SUBMISSION
* NAME: Xu Ziyin
* UID : 3036173372

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
struct mat_vec_mul_args{
    int no;
    float* out;
    float* vec;
    float* mat;
    int col;
    int start;
    int end;
    sem_t sem;
    int terminated;
    struct rusage thread_usage;
};

void *thr_func(void*) ;


pthread_t* threads;
struct mat_vec_mul_args* args;
int thread_count = 0;
sem_t sync;
int init_mat_vec_mul(int thr_count) {
    // a
    threads = malloc(thr_count * sizeof(pthread_t));
    args = malloc(thr_count * sizeof(struct mat_vec_mul_args));

    thread_count = thr_count; // init thread count
    sem_init(&sync, 0, 0);
    //b
    for(int i=0; i<thread_count; i++){
        sem_init(&(args[i].sem), 0, 0);
        args[i].terminated = 0;
        args[i].no = i;
        // c
        pthread_create(&threads[i], NULL, thr_func, &args[i]);
    }
    return 0;
}


void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    int line_for_each_thread = (row -1) / (thread_count) + 1;

    int remainder = row % thread_count;
    // a
    for(int i = 0; i < thread_count; i++) {
        struct mat_vec_mul_args* arg = &args[i];
        // *arg;
        arg->out = out;
        arg->vec = vec;
        arg->mat = mat;
        arg->col = col;
        arg->start = i * line_for_each_thread;
        arg->end = (i + 1) * line_for_each_thread;
        if (i == thread_count - 1) {
            arg->end = row;
        }
        // b
        sem_post(&(arg->sem));
    }
    // c
    for(int i=0;i<thread_count;i++){
        sem_wait(&sync);
    }
}


int close_mat_vec_mul() {
    // a
    for (int i = 0; i < thread_count; i++) {
        args[i].terminated = 1;
        sem_post(&(args[i].sem));
        
    }
    for(int i=0;i<thread_count;i++){
        sem_wait(&sync);
            printf("Thread %d has completed - user: %.4f s, system: %.4f s\n", i,
        (args[i].thread_usage.ru_utime.tv_sec + args[i].thread_usage.ru_utime.tv_usec/1000000.0),
        (args[i].thread_usage.ru_stime.tv_sec + args[i].thread_usage.ru_stime.tv_usec/1000000.0));
    }
    
    // b
    getrusage(RUSAGE_SELF, &main_usage);
    printf("main thread - user: %.4f s, system: %.4f s\n",
    (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec/1000000.0),
    (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec/1000000.0));

    // c
    for(int i=0; i<thread_count;i++){
        sem_destroy(&(args[i].sem));
    }
    free(args);
    sem_destroy(&sync);
    free(threads);
    return 0;
}


void *thr_func(void *arg) {
    struct mat_vec_mul_args* args = (struct mat_vec_mul_args*) arg;
    while(1){
        // a b
        sem_wait(&(args->sem));
        if(args->terminated){
            // d
            break;
        }
        int col = args->col;
        int start = args->start;
        int end = args->end;
        float* out = args->out;
        float* vec = args->vec;
        float* mat = args->mat;
        for (int i = start; i < end; i++) {
            float val = 0.0f;
            for(int j=0; j < col; j++){
                val += vec[j] * mat[i * col + j];
            }
            out[i] = val;
        }
        // c
        sem_post(&sync);
    }
    getrusage(RUSAGE_THREAD, &(args->thread_usage));
    sem_post(&sync);
    return NULL;
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
