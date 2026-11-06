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
#[command(about = "RustFS S3-over-RDMA GET and PUT over NIXL/UCX")]
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
    /// `get` reads the object into the buffer; `put` uploads the buffer as the
    /// object (multipart, the server pulls every part over NIXL); `put-objects`
    /// uploads `--count` objects of `--size` bytes as single PutObjects;
    /// `get-objects` reads them back over NIXL
    #[arg(long, default_value = "get")]
    op: String,
    /// PUT: object size in bytes
    #[arg(long, default_value_t = 2 << 30)]
    size: usize,
    /// put-objects: number of objects (keys `<key>-<index>`)
    #[arg(long, default_value_t = 1)]
    count: usize,
    /// PUT: multipart part size in bytes (96 MiB = two 12+4 stripes of 4 MiB shards)
    #[arg(long, default_value_t = 96 << 20)]
    part_size: usize,
    /// PUT: parts in flight; put-objects: objects in flight
    #[arg(long, default_value_t = 8)]
    parallel: usize,
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
            None => out.copy_from_slice(unsafe {
                std::slice::from_raw_parts(self.addr as *const u8, self.len)
            }),
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
    etag: Option<String>,
}

async fn request(
    client: &reqwest::Client,
    creds: &sigv4::Credentials,
    method: &str,
    url: &Url,
    extra: &[(String, String)],
) -> Result<(Reply, Vec<u8>)> {
    request_with_body(client, creds, method, url, extra, None).await
}

async fn request_with_body(
    client: &reqwest::Client,
    creds: &sigv4::Credentials,
    method: &str,
    url: &Url,
    extra: &[(String, String)],
    body: Option<Vec<u8>>,
) -> Result<(Reply, Vec<u8>)> {
    let headers = sigv4::sign(creds, method, url, extra, time::OffsetDateTime::now_utc());
    let mut req = client.request(method.parse()?, url.clone());
    for (name, value) in headers {
        req = req.header(name, value);
    }
    if let Some(body) = body {
        req = req.body(body);
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
    let etag = resp
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let body = resp.bytes().await.context("reading body")?.to_vec();
    let body_len = content_length.unwrap_or(body.len());
    Ok((
        Reply {
            status,
            rdma_reply,
            transferred,
            body_len,
            etag,
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
    if args.op == "put" {
        return run_put(&args, &client, &creds, &url).await;
    }
    if args.op == "put-objects" {
        return run_put_objects(&args, &client, &creds, &url).await;
    }
    if args.op == "get-objects" {
        return run_get_objects(&args, &client, &creds, &url).await;
    }

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
    let mut params =
        Params::from(std::iter::empty::<(&str, &str)>()).map_err(|e| anyhow!("params: {e}"))?;
    if !args.devices.is_empty() {
        params
            .set("device_list", &args.devices)
            .map_err(|e| anyhow!("device_list: {e}"))?;
    }
    let backend = agent
        .create_backend("UCX", &params)
        .map_err(|e| anyhow!("ucx backend: {e}"))?;
    let mut opt_args = OptArgs::new().map_err(|e| anyhow!("opt args: {e}"))?;
    opt_args
        .add_backend(&backend)
        .map_err(|e| anyhow!("opt args: {e}"))?;

    let buffer = match args.mem.as_str() {
        "gpu" | "vram" => Buffer::gpu(size, args.gpu)?,
        _ => Buffer::host(size)?,
    };
    let _registration = agent
        .register_memory(&buffer, Some(&opt_args))
        .map_err(|e| anyhow!("register memory: {e}"))?;
    let metadata = base64::engine::general_purpose::STANDARD
        .encode(agent.get_local_md().map_err(|e| anyhow!("metadata: {e}"))?);
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
            bail!(
                "the server did not serve the GET over RDMA (reply {:?}, moved {moved})",
                reply.rdma_reply
            );
        }
        last = Some(seconds);
        last_end = epoch();
    }

    let mut verified = None;
    if args.verify {
        let (reply, body) = request(&client, &creds, "GET", &url, &[]).await?;
        if reply.status != 200 || body.len() != size {
            bail!(
                "HTTP GET for verification returned {} with {} bytes",
                reply.status,
                body.len()
            );
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
                        first = Some(
                            index * BLOCK + a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0),
                        );
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
            // Provenance of the wrong blocks: which block of the object (if any) each one holds.
            let reference: std::collections::HashMap<[u8; 16], usize> = body
                .chunks(BLOCK)
                .enumerate()
                .map(|(index, block)| (Md5::digest(block).into(), index))
                .collect();
            let mut shown = 0usize;
            let mut from_object = 0usize;
            let mut unknown = 0usize;
            let mut listing = String::new();
            for (index, (a, b)) in ours.chunks(BLOCK).zip(body.chunks(BLOCK)).enumerate() {
                if a == b || a.iter().all(|&x| x == 0) {
                    continue;
                }
                let source = reference.get::<[u8; 16]>(&Md5::digest(a).into()).copied();
                match source {
                    Some(_) => from_object += 1,
                    None => unknown += 1,
                }
                if shown < 32 {
                    listing.push_str(&format!(
                        " {index}<-{}",
                        source.map_or("?".to_owned(), |s| s.to_string())
                    ));
                    shown += 1;
                }
            }
            println!("MISMATCH wrong_blocks_from_object={from_object} wrong_blocks_unknown={unknown} first:{listing}");
        }
        if !same {
            // Does the copy become correct later? Then the server replied before the data landed.
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let again = buffer.to_host()?;
            println!(
                "MISMATCH recheck_after_2s={}",
                if Md5::digest(&again) == Md5::digest(&body) {
                    "matches_now"
                } else {
                    "still_wrong"
                }
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

/// Fills `buf`, which starts `offset` bytes into the object, with the
/// object's deterministic pseudo-random content (splitmix64 of the word index).
fn fill_pattern(buf: &mut [u8], offset: usize) {
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    debug_assert!(offset % 8 == 0);
    for (k, word) in buf.chunks_mut(8).enumerate() {
        let value = mix(((offset / 8) + k) as u64).to_le_bytes();
        word.copy_from_slice(&value[..word.len()]);
    }
}

/// Fills the whole buffer with the object content, in parallel.
fn fill_buffer(buffer: &Buffer) -> Result<()> {
    const PIECE: usize = 64 << 20;
    let pieces: Vec<usize> = (0..buffer.len.div_ceil(PIECE)).collect();
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(32);
    match &buffer.cuda {
        None => {
            // SAFETY: host buffer of `len` bytes, exclusively ours; pieces are disjoint.
            let all = unsafe { std::slice::from_raw_parts_mut(buffer.addr as *mut u8, buffer.len) };
            let mut chunks: Vec<(usize, &mut [u8])> = all.chunks_mut(PIECE).enumerate().collect();
            let per_thread = chunks.len().div_ceil(threads).max(1);
            std::thread::scope(|scope| {
                for group in chunks.chunks_mut(per_thread) {
                    scope.spawn(move || {
                        for (index, chunk) in group.iter_mut() {
                            fill_pattern(chunk, *index * PIECE);
                        }
                    });
                }
            });
        }
        Some(cuda) => {
            let mut staging = vec![0u8; PIECE];
            for piece in pieces {
                let offset = piece * PIECE;
                let len = PIECE.min(buffer.len - offset);
                fill_pattern(&mut staging[..len], offset);
                cuda.copy_from_host(buffer.addr + offset, &staging[..len])?;
            }
            cuda.synchronize()?;
        }
    }
    Ok(())
}

fn xml_tag<'a>(body: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let start = body.find(&open)? + open.len();
    let end = body[start..].find(&format!("</{tag}>"))? + start;
    Some(&body[start..end])
}

/// PUT: uploads the registered buffer as a multipart object whose parts the
/// server pulls over NIXL (`x-rustfs-nixl-buffer` names each part's slice).
async fn run_put(
    args: &Args,
    client: &reqwest::Client,
    creds: &sigv4::Credentials,
    url: &Url,
) -> Result<()> {
    let size = args.size;
    if size == 0 || args.part_size == 0 {
        bail!("--size and --part-size must be positive");
    }
    let agent_name = if args.agent.is_empty() {
        format!("nixl-put-{}", std::process::id())
    } else {
        args.agent.clone()
    };
    let agent = Agent::new(&agent_name).map_err(|e| anyhow!("agent: {e}"))?;
    let mut params =
        Params::from(std::iter::empty::<(&str, &str)>()).map_err(|e| anyhow!("params: {e}"))?;
    if !args.devices.is_empty() {
        params
            .set("device_list", &args.devices)
            .map_err(|e| anyhow!("device_list: {e}"))?;
    }
    let backend = agent
        .create_backend("UCX", &params)
        .map_err(|e| anyhow!("ucx backend: {e}"))?;
    let mut opt_args = OptArgs::new().map_err(|e| anyhow!("opt args: {e}"))?;
    opt_args
        .add_backend(&backend)
        .map_err(|e| anyhow!("opt args: {e}"))?;
    let buffer = match args.mem.as_str() {
        "gpu" | "vram" => Buffer::gpu(size, args.gpu)?,
        _ => Buffer::host(size)?,
    };
    fill_buffer(&buffer)?;
    let _registration = agent
        .register_memory(&buffer, Some(&opt_args))
        .map_err(|e| anyhow!("register memory: {e}"))?;
    let metadata = base64::engine::general_purpose::STANDARD
        .encode(agent.get_local_md().map_err(|e| anyhow!("metadata: {e}"))?);
    let kind = if matches!(buffer.mem_type, MemType::Vram) {
        "vram"
    } else {
        "dram"
    };
    let parts = size.div_ceil(args.part_size);
    eprintln!(
        "object {size} bytes in {parts} parts, agent {agent_name}, buffer {}",
        buffer.header()
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
        let started = Instant::now();
        let mut create = url.clone();
        create.query_pairs_mut().append_pair("uploads", "");
        let create_str = create.as_str().trim_end_matches('=').to_owned();
        let create = Url::parse(&create_str)?;
        let (reply, body) = request(client, creds, "POST", &create, &[]).await?;
        if reply.status != 200 {
            bail!(
                "CreateMultipartUpload returned {}: {}",
                reply.status,
                String::from_utf8_lossy(&body)
            );
        }
        let body = String::from_utf8_lossy(&body).into_owned();
        let upload_id = xml_tag(&body, "UploadId")
            .context("no UploadId in CreateMultipartUpload reply")?
            .to_owned();

        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(args.parallel.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        for part in 0..parts {
            let permit = std::sync::Arc::clone(&semaphore).acquire_owned().await?;
            let offset = part * args.part_size;
            let len = args.part_size.min(size - offset);
            let mut part_url = url.clone();
            part_url
                .query_pairs_mut()
                .append_pair("partNumber", &(part + 1).to_string())
                .append_pair("uploadId", &upload_id);
            let extra = vec![
                ("x-rustfs-nixl-agent".to_owned(), agent_name.clone()),
                ("x-rustfs-nixl-metadata".to_owned(), metadata.clone()),
                (
                    "x-rustfs-nixl-buffer".to_owned(),
                    format!("{kind}:{}:{:#x}:{len}", buffer.device, buffer.addr + offset),
                ),
            ];
            let (client, creds) = (
                client.clone(),
                sigv4::Credentials {
                    access_key: creds.access_key.clone(),
                    secret_key: creds.secret_key.clone(),
                    region: creds.region.clone(),
                },
            );
            tasks.spawn(async move {
                let _permit = permit;
                let (reply, body) = request(&client, &creds, "PUT", &part_url, &extra).await?;
                if reply.status != 200
                    || reply.rdma_reply.as_deref() != Some("200")
                    || reply.transferred != Some(len as u64)
                {
                    bail!(
                        "UploadPart {} returned {} (rdma reply {:?}, moved {:?}): {}",
                        part + 1,
                        reply.status,
                        reply.rdma_reply,
                        reply.transferred,
                        String::from_utf8_lossy(&body)
                    );
                }
                let etag = reply.etag.context("UploadPart reply has no ETag")?;
                Ok::<_, anyhow::Error>((part + 1, etag))
            });
        }
        let mut etags = Vec::with_capacity(parts);
        while let Some(done) = tasks.join_next().await {
            etags.push(done??);
        }
        etags.sort();
        let mut xml = String::from("<CompleteMultipartUpload>");
        for (number, etag) in &etags {
            xml.push_str(&format!(
                "<Part><PartNumber>{number}</PartNumber><ETag>{etag}</ETag></Part>"
            ));
        }
        xml.push_str("</CompleteMultipartUpload>");
        let mut complete = url.clone();
        complete
            .query_pairs_mut()
            .append_pair("uploadId", &upload_id);
        let (reply, body) = request_with_body(
            client,
            creds,
            "POST",
            &complete,
            &[],
            Some(xml.into_bytes()),
        )
        .await?;
        if reply.status != 200 {
            bail!(
                "CompleteMultipartUpload returned {}: {}",
                reply.status,
                String::from_utf8_lossy(&body)
            );
        }
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "round {round}: parts={parts} seconds={seconds:.3} GBps={:.2}",
            size as f64 / seconds / 1e9
        );
        last = Some(seconds);
        last_end = epoch();
    }

    let mut verified = None;
    if args.verify {
        let (reply, body) = request(client, creds, "GET", url, &[]).await?;
        if reply.status != 200 || body.len() != size {
            bail!(
                "HTTP GET for verification returned {} with {} bytes",
                reply.status,
                body.len()
            );
        }
        let ours = buffer.to_host()?;
        let same = Md5::digest(&ours) == Md5::digest(&body);
        if !same {
            const BLOCK: usize = 1 << 20;
            let wrong = ours
                .chunks(BLOCK)
                .zip(body.chunks(BLOCK))
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "MISMATCH wrong_1MiB_blocks={wrong} of {}",
                size.div_ceil(BLOCK)
            );
        }
        verified = Some(same);
    }
    let seconds = last.unwrap_or(0.0);
    println!(
        "RESULT {} op=put size={size} seconds={seconds:.3} GBps={:.2} mem={} verified={} rounds={} start={first_start:.3} end={last_end:.3}",
        if verified != Some(false) { "OK" } else { "MISMATCH" },
        size as f64 / seconds / 1e9,
        args.mem,
        verified.map_or("skipped".to_owned(), |v| v.to_string()),
        args.rounds,
    );
    if verified == Some(false) {
        std::process::exit(2);
    }
    Ok(())
}

/// put-objects: uploads `--count` objects of `--size` bytes, each a single
/// PutObject whose body the server pulls over NIXL from its own slice of one
/// registered buffer, with `--parallel` objects in flight.
async fn run_put_objects(
    args: &Args,
    client: &reqwest::Client,
    creds: &sigv4::Credentials,
    url: &Url,
) -> Result<()> {
    let (size, count) = (args.size, args.count);
    if size == 0 || count == 0 || size % 8 != 0 {
        bail!("--size must be a positive multiple of 8 and --count positive");
    }
    let total = size * count;
    let agent_name = if args.agent.is_empty() {
        format!("nixl-puts-{}", std::process::id())
    } else {
        args.agent.clone()
    };
    let agent = Agent::new(&agent_name).map_err(|e| anyhow!("agent: {e}"))?;
    let mut params =
        Params::from(std::iter::empty::<(&str, &str)>()).map_err(|e| anyhow!("params: {e}"))?;
    if !args.devices.is_empty() {
        params
            .set("device_list", &args.devices)
            .map_err(|e| anyhow!("device_list: {e}"))?;
    }
    let backend = agent
        .create_backend("UCX", &params)
        .map_err(|e| anyhow!("ucx backend: {e}"))?;
    let mut opt_args = OptArgs::new().map_err(|e| anyhow!("opt args: {e}"))?;
    opt_args
        .add_backend(&backend)
        .map_err(|e| anyhow!("opt args: {e}"))?;
    let buffer = match args.mem.as_str() {
        "gpu" | "vram" => Buffer::gpu(total, args.gpu)?,
        _ => Buffer::host(total)?,
    };
    fill_buffer(&buffer)?;
    let _registration = agent
        .register_memory(&buffer, Some(&opt_args))
        .map_err(|e| anyhow!("register memory: {e}"))?;
    let metadata = base64::engine::general_purpose::STANDARD
        .encode(agent.get_local_md().map_err(|e| anyhow!("metadata: {e}"))?);
    let kind = if matches!(buffer.mem_type, MemType::Vram) {
        "vram"
    } else {
        "dram"
    };
    let object_url = |index: usize| -> Result<Url> {
        let base = url.as_str().trim_end_matches('/');
        Ok(Url::parse(&format!("{base}-{index:05}"))?)
    };
    eprintln!(
        "{count} objects of {size} bytes, {} in flight, agent {agent_name}, buffer {}",
        args.parallel,
        buffer.header()
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
    let mut latencies_ms = Vec::with_capacity(count * args.rounds);
    let mut etags: Vec<Option<String>> = vec![None; count];
    for round in 0..args.rounds {
        let started = Instant::now();
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(args.parallel.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        for index in 0..count {
            let permit = std::sync::Arc::clone(&semaphore).acquire_owned().await?;
            let target = object_url(index)?;
            let extra = vec![
                ("x-rustfs-nixl-agent".to_owned(), agent_name.clone()),
                ("x-rustfs-nixl-metadata".to_owned(), metadata.clone()),
                (
                    "x-rustfs-nixl-buffer".to_owned(),
                    format!(
                        "{kind}:{}:{:#x}:{size}",
                        buffer.device,
                        buffer.addr + index * size
                    ),
                ),
            ];
            let (client, creds) = (
                client.clone(),
                sigv4::Credentials {
                    access_key: creds.access_key.clone(),
                    secret_key: creds.secret_key.clone(),
                    region: creds.region.clone(),
                },
            );
            tasks.spawn(async move {
                let _permit = permit;
                let object_started = Instant::now();
                let (reply, body) = request(&client, &creds, "PUT", &target, &extra).await?;
                if reply.status != 200
                    || reply.rdma_reply.as_deref() != Some("200")
                    || reply.transferred != Some(size as u64)
                {
                    bail!(
                        "PutObject {index} returned {} (rdma reply {:?}, moved {:?}): {}",
                        reply.status,
                        reply.rdma_reply,
                        reply.transferred,
                        String::from_utf8_lossy(&body)
                    );
                }
                Ok::<_, anyhow::Error>((
                    index,
                    object_started.elapsed().as_secs_f64() * 1000.0,
                    reply.etag,
                ))
            });
        }
        while let Some(done) = tasks.join_next().await {
            let (index, latency_ms, etag) = done??;
            latencies_ms.push(latency_ms);
            etags[index] = etag;
        }
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "round {round}: objects={count} seconds={seconds:.3} GBps={:.2} objects_per_s={:.0}",
            total as f64 / seconds / 1e9,
            count as f64 / seconds
        );
        last = Some(seconds);
        last_end = epoch();
    }

    let mut verified = None;
    if args.verify {
        let ours = buffer.to_host()?;
        let mut same = true;
        let mut tasks = tokio::task::JoinSet::new();
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(args.parallel.max(1)));
        for index in 0..count {
            let permit = std::sync::Arc::clone(&semaphore).acquire_owned().await?;
            let target = object_url(index)?;
            let (client, creds) = (
                client.clone(),
                sigv4::Credentials {
                    access_key: creds.access_key.clone(),
                    secret_key: creds.secret_key.clone(),
                    region: creds.region.clone(),
                },
            );
            tasks.spawn(async move {
                let _permit = permit;
                let (reply, body) = request(&client, &creds, "GET", &target, &[]).await?;
                Ok::<_, anyhow::Error>((index, reply.status, body))
            });
        }
        let mut wrong = 0usize;
        let mut wrong_etags = 0usize;
        while let Some(done) = tasks.join_next().await {
            let (index, status, body) = done??;
            let slice = &ours[index * size..(index + 1) * size];
            if status != 200 || body.as_slice() != slice {
                wrong += 1;
                same = false;
            }
            // The PUT's ETag must be the MD5 of the object's bytes.
            let expected = format!("\"{}\"", hex_string(&Md5::digest(slice)));
            if etags[index].as_deref() != Some(expected.as_str()) {
                wrong_etags += 1;
                same = false;
            }
        }
        if !same {
            println!("MISMATCH wrong_objects={wrong} wrong_etags={wrong_etags} of {count}");
        }
        verified = Some(same);
    }
    latencies_ms.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        latencies_ms
            .get(((latencies_ms.len() as f64 - 1.0) * p).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let seconds = last.unwrap_or(0.0);
    println!(
        "RESULT {} op=put-objects size={total} objects={count} object_size={size} seconds={seconds:.3} GBps={:.2} objects_per_s={:.0} p50_ms={:.1} p99_ms={:.1} mem={} verified={} rounds={} start={first_start:.3} end={last_end:.3}",
        if verified != Some(false) { "OK" } else { "MISMATCH" },
        total as f64 / seconds / 1e9,
        count as f64 / seconds,
        percentile(0.5),
        percentile(0.99),
        args.mem,
        verified.map_or("skipped".to_owned(), |v| v.to_string()),
        args.rounds,
    );
    if verified == Some(false) {
        std::process::exit(2);
    }
    Ok(())
}

/// get-objects: reads the `--count` objects `put-objects` wrote (`<key>-<index>`,
/// `--size` bytes each) over NIXL into their slices of one registered buffer,
/// `--parallel` in flight; `--verify` checks every slice against the content
/// `put-objects` wrote.
async fn run_get_objects(
    args: &Args,
    client: &reqwest::Client,
    creds: &sigv4::Credentials,
    url: &Url,
) -> Result<()> {
    let (size, count) = (args.size, args.count);
    if size == 0 || count == 0 || size % 8 != 0 {
        bail!("--size must be a positive multiple of 8 and --count positive");
    }
    let total = size * count;
    let agent_name = if args.agent.is_empty() {
        format!("nixl-gets-{}", std::process::id())
    } else {
        args.agent.clone()
    };
    let agent = Agent::new(&agent_name).map_err(|e| anyhow!("agent: {e}"))?;
    let mut params =
        Params::from(std::iter::empty::<(&str, &str)>()).map_err(|e| anyhow!("params: {e}"))?;
    if !args.devices.is_empty() {
        params
            .set("device_list", &args.devices)
            .map_err(|e| anyhow!("device_list: {e}"))?;
    }
    let backend = agent
        .create_backend("UCX", &params)
        .map_err(|e| anyhow!("ucx backend: {e}"))?;
    let mut opt_args = OptArgs::new().map_err(|e| anyhow!("opt args: {e}"))?;
    opt_args
        .add_backend(&backend)
        .map_err(|e| anyhow!("opt args: {e}"))?;
    let buffer = match args.mem.as_str() {
        "gpu" | "vram" => Buffer::gpu(total, args.gpu)?,
        _ => Buffer::host(total)?,
    };
    let _registration = agent
        .register_memory(&buffer, Some(&opt_args))
        .map_err(|e| anyhow!("register memory: {e}"))?;
    let metadata = base64::engine::general_purpose::STANDARD
        .encode(agent.get_local_md().map_err(|e| anyhow!("metadata: {e}"))?);
    let kind = if matches!(buffer.mem_type, MemType::Vram) {
        "vram"
    } else {
        "dram"
    };
    let object_url = |index: usize| -> Result<Url> {
        let base = url.as_str().trim_end_matches('/');
        Ok(Url::parse(&format!("{base}-{index:05}"))?)
    };
    eprintln!(
        "{count} objects of {size} bytes, {} in flight, agent {agent_name}, buffer {}",
        args.parallel,
        buffer.header()
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
    let mut latencies_ms = Vec::with_capacity(count * args.rounds);
    for round in 0..args.rounds {
        let started = Instant::now();
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(args.parallel.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        for index in 0..count {
            let permit = std::sync::Arc::clone(&semaphore).acquire_owned().await?;
            let target = object_url(index)?;
            let extra = vec![
                ("x-rustfs-nixl-agent".to_owned(), agent_name.clone()),
                ("x-rustfs-nixl-metadata".to_owned(), metadata.clone()),
                (
                    "x-rustfs-nixl-buffer".to_owned(),
                    format!(
                        "{kind}:{}:{:#x}:{size}",
                        buffer.device,
                        buffer.addr + index * size
                    ),
                ),
            ];
            let (client, creds) = (
                client.clone(),
                sigv4::Credentials {
                    access_key: creds.access_key.clone(),
                    secret_key: creds.secret_key.clone(),
                    region: creds.region.clone(),
                },
            );
            tasks.spawn(async move {
                let _permit = permit;
                let object_started = Instant::now();
                let (reply, body) = request(&client, &creds, "GET", &target, &extra).await?;
                if reply.status != 200
                    || reply.rdma_reply.is_none()
                    || reply.transferred != Some(size as u64)
                {
                    bail!(
                        "GetObject {index} returned {} (rdma reply {:?}, moved {:?}): {}",
                        reply.status,
                        reply.rdma_reply,
                        reply.transferred,
                        String::from_utf8_lossy(&body)
                    );
                }
                Ok::<_, anyhow::Error>(object_started.elapsed().as_secs_f64() * 1000.0)
            });
        }
        while let Some(done) = tasks.join_next().await {
            latencies_ms.push(done??);
        }
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "round {round}: objects={count} seconds={seconds:.3} GBps={:.2} objects_per_s={:.0}",
            total as f64 / seconds / 1e9,
            count as f64 / seconds
        );
        last = Some(seconds);
        last_end = epoch();
    }

    let mut verified = None;
    if args.verify {
        let ours = buffer.to_host()?;
        let mut expected = vec![0u8; size];
        let mut wrong = 0usize;
        for index in 0..count {
            fill_pattern(&mut expected, index * size);
            if ours[index * size..(index + 1) * size] != expected[..] {
                wrong += 1;
            }
        }
        if wrong > 0 {
            println!("MISMATCH wrong_objects={wrong} of {count}");
        }
        verified = Some(wrong == 0);
    }
    latencies_ms.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        latencies_ms
            .get(((latencies_ms.len() as f64 - 1.0) * p).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let seconds = last.unwrap_or(0.0);
    println!(
        "RESULT {} op=get-objects size={total} objects={count} object_size={size} seconds={seconds:.3} GBps={:.2} objects_per_s={:.0} p50_ms={:.1} p99_ms={:.1} mem={} verified={} rounds={} start={first_start:.3} end={last_end:.3}",
        if verified != Some(false) { "OK" } else { "MISMATCH" },
        total as f64 / seconds / 1e9,
        count as f64 / seconds,
        percentile(0.5),
        percentile(0.99),
        args.mem,
        verified.map_or("skipped".to_owned(), |v| v.to_string()),
        args.rounds,
    );
    if verified == Some(false) {
        std::process::exit(2);
    }
    Ok(())
}

/// Lower-case hex of `bytes`.
fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
