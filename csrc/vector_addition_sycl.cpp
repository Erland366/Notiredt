using namespace sycl;

void vector_add(float *A, float *B, float *C, int n){
    buffer<float> bufA(A, n);
    buffer<float> bufB(B, n);
    buffer<float> bufC(C, n);

    // SYCL Queue
    queue q;
    q.submit([&](handler &h){
        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);

        h.parallel_for(n, [=](id<1> i){
            accC[i] = accA[i] + accB[i];
        });
    });

    q.wait();
}