void solveCPU(float *average, const float* const input, const int size) {
    for (int i = 0; i < size; i++) {
        float avg = 0.0f;
        for (int w = -R; w < R; w++) {
            int idx = i + w;
            if (idx < 0) idx = 0;
            if (idx >= size) idx = size-1;
            avg += input[idx];
        }
        average[i] = avg/float(2*R);
    }
}
