// gdr_smoke.c — verify GDRCopy: cudaMalloc 16MB GPU mem, pin via gdr_pin_buffer,
// map to host VA, std::memcpy 16KB pattern, gdr_copy_to_mapping to flush, then
// read back via cudaMemcpy D→H and verify bytes match. Exits 0 on success.
//
// Build:  gcc gdr_smoke.c -o gdr_smoke \
//          -I/usr/local/gdrcopy/include -I/usr/local/cuda/include \
//          -L/usr/local/gdrcopy/lib -L/usr/local/cuda/lib64 \
//          -lgdrapi -lcudart
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gdrapi.h>

#define SZ (16 * 1024 * 1024)
#define MSG (16 * 1024)

static void die(const char* what, int rc) {
    fprintf(stderr, "FAIL: %s rc=%d\n", what, rc);
    exit(1);
}

int main(void) {
    cudaError_t ce;
    int rc;
    void* d = NULL;
    ce = cudaMalloc(&d, SZ); if (ce) die("cudaMalloc", ce);
    fprintf(stderr, "ok cudaMalloc=%p size=%d\n", d, SZ);

    gdr_t g = gdr_open();
    if (!g) die("gdr_open returned NULL", -1);
    fprintf(stderr, "ok gdr_open\n");

    gdr_mh_t mh;
    rc = gdr_pin_buffer(g, (CUdeviceptr)(uintptr_t)d, SZ, 0, 0, &mh);
    if (rc) die("gdr_pin_buffer", rc);
    fprintf(stderr, "ok gdr_pin_buffer\n");

    void* host_map = NULL;
    rc = gdr_map(g, mh, &host_map, SZ);
    if (rc) die("gdr_map", rc);
    fprintf(stderr, "ok gdr_map host_map=%p\n", host_map);

    gdr_info_t info;
    rc = gdr_get_info(g, mh, &info);
    if (rc) die("gdr_get_info", rc);
    fprintf(stderr, "ok gdr_get_info va=0x%lx mapped_size=%lu page_size=%u\n",
            (unsigned long)info.va, (unsigned long)info.mapped_size, info.page_size);

    // adjust offset (gdr_map may align to GPU page boundary)
    size_t off = ((uintptr_t)d) - info.va;
    fprintf(stderr, "ok offset_adjust=%zu\n", off);

    // write 16KB pattern at host_map+off via memcpy (with required gdr_copy helper)
    static char pattern[MSG];
    for (int i = 0; i < MSG; ++i) pattern[i] = (char)(i & 0xff);

    rc = gdr_copy_to_mapping(mh, (char*)host_map + off, pattern, MSG);
    if (rc) die("gdr_copy_to_mapping", rc);
    fprintf(stderr, "ok gdr_copy_to_mapping 16KB\n");

    // Read back from GPU via cudaMemcpy
    static char readback[MSG];
    ce = cudaMemcpy(readback, d, MSG, cudaMemcpyDeviceToHost);
    if (ce) die("cudaMemcpy D->H", ce);
    fprintf(stderr, "ok cudaMemcpy D->H\n");

    if (memcmp(readback, pattern, MSG) != 0) {
        for (int i = 0; i < 16; ++i) {
            fprintf(stderr, " readback[%d]=0x%02x pattern[%d]=0x%02x\n",
                    i, (unsigned)readback[i] & 0xff, i, (unsigned)pattern[i] & 0xff);
        }
        die("memcmp mismatch", 1);
    }
    fprintf(stderr, "PASS: GPU mem readback matches pattern\n");

    gdr_unmap(g, mh, host_map, SZ);
    gdr_unpin_buffer(g, mh);
    gdr_close(g);
    cudaFree(d);
    return 0;
}
