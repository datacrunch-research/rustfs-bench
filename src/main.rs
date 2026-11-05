//! `nixl-get`: fetch an object from RustFS over NIXL/UCX into GPU or host
//! memory and report the transfer rate.
//!
//! The GET carries the client's NIXL agent name, agent metadata and the
//! registered buffer (`x-rustfs-nixl-*` headers); the server moves the bytes
//! with NIXL WRITE transfers and answers with an empty body plus
//! `x-amz-rdma-reply` and `x-amz-rdma-bytes-transferred`. With `--verify`
//! the object is fetched once more over plain HTTP and compared byte for
//! byte with the RDMA copy.

mod cuda;
mod sigv4;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use clap::Parser;
use md5::{Digest, Md5};
use nixl_sys::{Agent, MemType, MemoryRegion, NixlDescriptor, OptArgs, Params};
use std::alloc::Layout;
use std::time::Instant;
use url::Url;

#[derive(Parser, Debug)]
#[command(about = "RustFS S3-over-RDMA GET over NIXL/UCX")]
struct Args {
    /// S3 endpoint, for example http://10.20.9.57:9000
    #[arg(long)]
    endpoint: String,
    #[arg(long)]
    bucket: String,
    #[arg(long)]
    key: String,
    /// Where the object lands: host or gpu
    #[arg(long, default_value = "host")]
    mem: String,
    /// CUDA device index for --mem gpu (also the NIXL device id)
    #[arg(long, default_value_t = 0)]
    gpu: i32,
    /// UCX device names for the agent, for example mlx5_9 (empty: every device)
    #[arg(long, default_value = "")]
    devices: String,
    /// This client's NIXL agent name (unique per process)
    #[arg(long, default_value = "")]
    agent: String,
    /// Repeat the GET this many times into the same buffer
    #[arg(long, default_value_t = 1)]
    rounds: usize,
    /// Fetch the object over HTTP afterwards and compare it with the RDMA copy
    #[arg(long)]
    verify: bool,
    /// Send the agent metadata on every round (default: first round only)
    #[arg(long)]
    metadata_every_round: bool,
    /// Wait until this Unix time (seconds) before the first round, so several
    /// processes start together and their wall clocks can be compared
    #[arg(long)]
    start_at: Option<f64>,
    #[arg(long, default_value = "rustfsadmin")]
    access_key: String,
    #[arg(long, default_value = "rustfsadmin")]
    secret_key: String,
    #[arg(long, default_value = "us-east-1")]
    region: String,
}

/// The receive buffer: page-aligned host memory or a CUDA allocation.
struct Buffer {
    addr: usize,
    len: usize,
    mem_type: MemType,
    device: u64,
    host_layout: Option<Layout>,
    cuda: Option<cuda::Cuda>,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("addr", &format_args!("{:#x}", self.addr))
            .field("len", &self.len)
            .field("mem_type", &self.mem_type)
            .finish()
    }
}

impl Buffer {
    fn host(len: usize) -> Result<Self> {
        let layout = Layout::from_size_align(len, 4096)?;
        // SAFETY: non-zero size; freed in Drop with the same layout.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            bail!("host allocation of {len} bytes failed");
        }
        Ok(Self {
            addr: ptr as usize,
            len,
            mem_type: MemType::Dram,
            device: 0,
            host_layout: Some(layout),
            cuda: None,
        })
    }

    fn gpu(len: usize, device: i32) -> Result<Self> {
        let cuda = cuda::Cuda::load()?;
        cuda.set_device(device)?;
        let addr = cuda.malloc(len)?;
        cuda.memset(addr, 0, len)?;
        Ok(Self {
            addr,
            len,
            mem_type: MemType::Vram,
            device: u64::try_from(device).unwrap_or(0),
            host_layout: None,
            cuda: Some(cuda),
        })
    }

    fn header(&self) -> String {
        let kind = match self.mem_type {
            MemType::Vram => "vram",
            _ => "dram",
        };
        format!("{kind}:{}:{:#x}:{}", self.device, self.addr, self.len)
    }

    /// A host copy of the buffer contents for verification.
    fn to_host(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.len];
        match &self.cuda {
            Some(cuda) => cuda.copy_to_host(&mut out, self.addr)?,
            // SAFETY: host buffer of `len` bytes, exclusively ours.
            None => out.copy_from_slice(unsafe { std::slice::from_raw_parts(self.addr as *const u8, self.len) }),
        }
        Ok(out)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(layout) = self.host_layout {
            // SAFETY: allocated in `host` with this layout.
            unsafe { std::alloc::dealloc(self.addr as *mut u8, layout) };
        }
    }
}

impl MemoryRegion for Buffer {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }

    fn size(&self) -> usize {
        self.len
    }
}

impl NixlDescriptor for Buffer {
    fn mem_type(&self) -> MemType {
        self.mem_type
    }

    fn device_id(&self) -> u64 {
        self.device
    }
}

struct Reply {
    status: u16,
    rdma_reply: Option<String>,
    transferred: Option<u64>,
    body_len: usize,
}

async fn request(
    client: &reqwest::Client,
    creds: &sigv4::Credentials,
    method: &str,
    url: &Url,
    extra: &[(String, String)],
) -> Result<(Reply, Vec<u8>)> {
    let headers = sigv4::sign(creds, method, url, extra, time::OffsetDateTime::now_utc());
    let mut req = client.request(method.parse()?, url.clone());
    for (name, value) in headers {
        req = req.header(name, value);
    }
    let resp = req.send().await.context("request failed")?;
    let status = resp.status().as_u16();
    let rdma_reply = resp
        .headers()
        .get("x-amz-rdma-reply")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let transferred = resp
        .headers()
        .get("x-amz-rdma-bytes-transferred")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());
    let content_length = resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    let body = resp.bytes().await.context("reading body")?.to_vec();
    let body_len = content_length.unwrap_or(body.len());
    Ok((
        Reply {
            status,
            rdma_reply,
            transferred,
            body_len,
        },
        body,
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let creds = sigv4::Credentials {
        access_key: args.access_key.clone(),
        secret_key: args.secret_key.clone(),
        region: args.region.clone(),
    };
    let base = Url::parse(&args.endpoint)?;
    let url = base.join(&format!("{}/{}", args.bucket, args.key))?;
    let client = reqwest::Client::builder().build()?;

    let (head, _) = request(&client, &creds, "HEAD", &url, &[]).await?;
    if head.status != 200 {
        bail!("HEAD {} returned {}", url, head.status);
    }
    let size = head.body_len;
    if size == 0 {
        bail!("HEAD returned no Content-Length or an empty object");
    }

    let agent_name = if args.agent.is_empty() {
        format!("nixl-get-{}", std::process::id())
    } else {
        args.agent.clone()
    };
    let agent = Agent::new(&agent_name).map_err(|e| anyhow!("agent: {e}"))?;
    let mut params = Params::from(std::iter::empty::<(&str, &str)>()).map_err(|e| anyhow!("params: {e}"))?;
    if !args.devices.is_empty() {
        params.set("device_list", &args.devices).map_err(|e| anyhow!("device_list: {e}"))?;
    }
    let backend = agent.create_backend("UCX", &params).map_err(|e| anyhow!("ucx backend: {e}"))?;
    let mut opt_args = OptArgs::new().map_err(|e| anyhow!("opt args: {e}"))?;
    opt_args.add_backend(&backend).map_err(|e| anyhow!("opt args: {e}"))?;

    let buffer = match args.mem.as_str() {
        "gpu" | "vram" => Buffer::gpu(size, args.gpu)?,
        _ => Buffer::host(size)?,
    };
    let _registration = agent
        .register_memory(&buffer, Some(&opt_args))
        .map_err(|e| anyhow!("register memory: {e}"))?;
    let metadata = base64::engine::general_purpose::STANDARD.encode(agent.get_local_md().map_err(|e| anyhow!("metadata: {e}"))?);
    eprintln!(
        "object {} bytes, agent {agent_name}, buffer {}, metadata {} bytes",
        size,
        buffer.header(),
        metadata.len()
    );

    if let Some(start_at) = args.start_at {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        if start_at > now {
            tokio::time::sleep(std::time::Duration::from_secs_f64(start_at - now)).await;
        }
    }
    let epoch = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    };
    let first_start = epoch();
    let mut last_end = first_start;
    let mut last = None;
    for round in 0..args.rounds {
        let mut extra = vec![
            ("x-rustfs-nixl-agent".to_owned(), agent_name.clone()),
            ("x-rustfs-nixl-buffer".to_owned(), buffer.header()),
        ];
        if round == 0 || args.metadata_every_round {
            extra.push(("x-rustfs-nixl-metadata".to_owned(), metadata.clone()));
        }
        let started = Instant::now();
        let (reply, _) = request(&client, &creds, "GET", &url, &extra).await?;
        let seconds = started.elapsed().as_secs_f64();
        let moved = reply.transferred.unwrap_or(0);
        println!(
            "round {round}: status={} reply={} transferred={moved} body_len={} seconds={seconds:.3} GBps={:.2}",
            reply.status,
            reply.rdma_reply.as_deref().unwrap_or("-"),
            reply.body_len,
            moved as f64 / seconds / 1e9
        );
        if reply.rdma_reply.is_none() || moved != size as u64 {
            bail!("the server did not serve the GET over RDMA (reply {:?}, moved {moved})", reply.rdma_reply);
        }
        last = Some(seconds);
        last_end = epoch();
    }

    let mut verified = None;
    if args.verify {
        let (reply, body) = request(&client, &creds, "GET", &url, &[]).await?;
        if reply.status != 200 || body.len() != size {
            bail!("HTTP GET for verification returned {} with {} bytes", reply.status, body.len());
        }
        let ours = buffer.to_host()?;
        let same = Md5::digest(&ours) == Md5::digest(&body);
        if !same {
            // Where and how the copies differ: never-written (still zero) or stale/wrong bytes, per 1 MiB block.
            const BLOCK: usize = 1 << 20;
            let mut first = None;
            let mut zero_blocks = 0usize;
            let mut wrong_blocks = 0usize;
            for (index, (a, b)) in ours.chunks(BLOCK).zip(body.chunks(BLOCK)).enumerate() {
                if a != b {
                    if first.is_none() {
                        first = Some(index * BLOCK + a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0));
                    }
                    if a.iter().all(|&x| x == 0) {
                        zero_blocks += 1;
                    } else {
                        wrong_blocks += 1;
                    }
                }
            }
            println!(
                "MISMATCH first_offset={:?} zero_1MiB_blocks={zero_blocks} wrong_1MiB_blocks={wrong_blocks} of {}",
                first,
                ours.len().div_ceil(BLOCK)
            );
        }
        if !same {
            // Does the copy become correct later? Then the server replied before the data landed.
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let again = buffer.to_host()?;
            println!(
                "MISMATCH recheck_after_2s={}",
                if Md5::digest(&again) == Md5::digest(&body) { "matches_now" } else { "still_wrong" }
            );
        }
        verified = Some(same);
    }
    let seconds = last.unwrap_or(0.0);
    println!(
        "RESULT {} size={size} seconds={seconds:.3} GBps={:.2} mem={} verified={} rounds={} start={first_start:.3} end={:.3}",
        if verified != Some(false) { "OK" } else { "MISMATCH" },
        size as f64 / seconds / 1e9,
        args.mem,
        verified.map_or("skipped".to_owned(), |v| v.to_string()),
        args.rounds,
        last_end
    );
    if verified == Some(false) {
        std::process::exit(2);
    }
    Ok(())
}
