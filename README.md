# rustfs-bench

Client-side benchmarks for RustFS S3-over-RDMA on NIXL/UCX.

## nixl-get

Fetches one object into GPU or host memory over NIXL (UCX backend) and
reports the transfer rate. The GET carries the client's NIXL agent name,
agent metadata and registered buffer in `x-rustfs-nixl-*` headers; the server
moves the bytes with NIXL WRITE transfers and answers with an empty body plus
`x-amz-rdma-reply` and `x-amz-rdma-bytes-transferred`. `--verify` fetches the
object again over plain HTTP and compares it byte for byte.

```bash
# build against a NIXL prefix (headers + libnixl); bindgen needs libclang
NIXL_PREFIX=/opt/nvidia/nvda_nixl cargo build --release

# run: the NIXL libraries and the UCX plugin must be on the library path
export LD_LIBRARY_PATH=$NIXL_PREFIX/lib:/usr/local/cuda/lib64 NIXL_PLUGIN_DIR=$NIXL_PREFIX/lib/plugins
nixl-get --endpoint http://10.20.9.57:9000 --bucket models --key bench/rand-2g-s4m.bin \
  --mem gpu --gpu 4 --devices mlx5_9 --rounds 3 --verify
```

`--devices` names the UCX devices (the NIC next to the target GPU); `--mem host`
lands in page-aligned host memory. Requires a RustFS server built with the
`nixl` feature and `RUSTFS_NIXL_ENABLE=true`.

## nixl-reuse

A standalone NIXL staging-chunk reuse test with no RustFS in the loop. `target`
registers a zeroed host buffer and publishes its agent metadata through a shared
directory; `writer` fills the buffer with 16 MiB WRITE transfers from a small
pool of chunks, refilling each chunk as soon as NIXL reports its transfer
complete, as the RustFS emit path does. The target then checks every 8-byte
word against its own offset and lists which offsets' data landed in the wrong
place.

```bash
# on the receiving node
nixl-reuse --id r1 --devices mlx5_11 target --size 2147483648
# on the sending node
nixl-reuse --id r1 --devices mlx5_11 writer --pool 8 --threads 8
```

Note that `Agent::post_xfer_req` in the `nixl-sys` binding returns `true`
while the transfer is still in progress and `false` when it completed before
returning; reading it the other way reports every transfer complete at post
time, which this test was written to catch.

