// Copyright 2025 RustFS Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Standalone NIXL staging-chunk reuse test, no RustFS involved.
//!
//! `target` registers a zeroed host buffer and publishes its agent metadata
//! and address through a shared directory; `writer` loads it and fills the
//! whole buffer with 16 MiB WRITE transfers from a small pool of staging
//! chunks, refilling a chunk the moment NIXL reports its transfer complete,
//! exactly as the RustFS emit path does. The target then checks every 8-byte
//! word against its own offset and reports which offsets' data landed in the
//! wrong place.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use nixl_sys::{Agent, AgentConfig, MemType, MemoryRegion, NixlDescriptor, OptArgs, Params, ThreadSync, XferDescList, XferOp, XferStatus};
use std::alloc::Layout;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
struct Args {
    /// Shared directory both sides can read and write
    #[arg(long, default_value = "/shared/nixl-reuse")]
    dir: PathBuf,
    /// Run id, names the exchange files
    #[arg(long, default_value = "r1")]
    id: String,
    /// UCX device names, comma-separated
    #[arg(long, default_value = "")]
    devices: String,
    #[command(subcommand)]
    role: Role,
}

#[derive(Subcommand, Debug)]
enum Role {
    /// Register the destination buffer, wait for the writer, verify
    Target {
        #[arg(long, default_value_t = 2 << 30)]
        size: usize,
    },
    /// Fill the target buffer with chunked WRITE transfers
    Writer {
        #[arg(long, default_value_t = 16 << 20)]
        chunk: usize,
        /// Staging chunks in the pool (and transfers in flight at most)
        #[arg(long, default_value_t = 8)]
        pool: usize,
        /// Worker threads sharing the agent; each owns pool/threads chunks
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Agent synchronization: none, rw, strict
        #[arg(long, default_value = "rw")]
        sync: String,
        #[arg(long, default_value_t = 1)]
        num_workers: u32,
        /// Write the buffer this many times
        #[arg(long, default_value_t = 1)]
        rounds: usize,
        /// Microseconds to wait after a completion before refilling the chunk
        #[arg(long, default_value_t = 0)]
        refill_delay_us: u64,
        /// Before the test, probe the endpoint until a transfer provably moved data:
        /// `read` (read the target's first page twice into differently filled
        /// chunks until both reads agree) or `write` (write a pattern, read it back)
        #[arg(long, default_value = "")]
        probe: String,
        /// Let NIXL run its own UCX progress thread
        #[arg(long, default_value_t = false)]
        prog_thread: bool,
    },
}

#[derive(Debug)]
struct HostBuffer {
    addr: usize,
    len: usize,
    layout: Layout,
}

impl HostBuffer {
    fn new(len: usize) -> Result<Self> {
        let layout = Layout::from_size_align(len, 4096)?;
        // SAFETY: non-zero size; freed in Drop with the same layout.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            bail!("host allocation of {len} bytes failed");
        }
        Ok(Self { addr: ptr as usize, len, layout })
    }

    fn slice_mut(&self, offset: usize, len: usize) -> &mut [u8] {
        assert!(offset + len <= self.len);
        // SAFETY: inside the allocation; callers hand out disjoint ranges.
        unsafe { std::slice::from_raw_parts_mut((self.addr + offset) as *mut u8, len) }
    }

    fn slice(&self, offset: usize, len: usize) -> &[u8] {
        assert!(offset + len <= self.len);
        // SAFETY: inside the allocation.
        unsafe { std::slice::from_raw_parts((self.addr + offset) as *const u8, len) }
    }
}

unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.addr as *mut u8, self.layout) };
    }
}

impl MemoryRegion for HostBuffer {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }

    fn size(&self) -> usize {
        self.len
    }
}

impl NixlDescriptor for HostBuffer {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

fn fill_pattern(buf: &mut [u8], offset: usize) {
    for (k, word) in buf.chunks_exact_mut(8).enumerate() {
        word.copy_from_slice(&((offset + k * 8) as u64).to_le_bytes());
    }
}

fn make_agent(name: &str, devices: &str, sync: ThreadSync, num_workers: u32, prog_thread: bool) -> Result<(Agent, OptArgs)> {
    let config = AgentConfig { thread_sync: sync, num_workers, enable_prog_thread: prog_thread, ..AgentConfig::default() };
    let agent = Agent::new_configured(name, &config).map_err(|e| anyhow!("agent: {e}"))?;
    let mut params = Params::from(std::iter::empty::<(&str, &str)>()).map_err(|e| anyhow!("params: {e}"))?;
    if !devices.is_empty() {
        params.set("device_list", devices).map_err(|e| anyhow!("device_list: {e}"))?;
    }
    let backend = agent.create_backend("UCX", &params).map_err(|e| anyhow!("ucx backend: {e}"))?;
    let mut opt_args = OptArgs::new().map_err(|e| anyhow!("opt args: {e}"))?;
    opt_args.add_backend(&backend).map_err(|e| anyhow!("opt args: {e}"))?;
    Ok((agent, opt_args))
}

fn target(args: &Args, size: usize) -> Result<()> {
    let name = format!("reuse-target-{}", std::process::id());
    // The target never polls, so NIXL's progress thread answers the writer's
    // connection handshake (the `nixl-get` client gets the same from `Agent::new`).
    let (agent, opt_args) = make_agent(&name, &args.devices, ThreadSync::Rw, 1, true)?;
    let buffer = HostBuffer::new(size)?;
    let _registration = agent.register_memory(&buffer, Some(&opt_args)).map_err(|e| anyhow!("register: {e}"))?;
    let metadata = agent.get_local_md().map_err(|e| anyhow!("metadata: {e}"))?;
    std::fs::create_dir_all(&args.dir)?;
    let base = args.dir.join(&args.id);
    let _ = std::fs::remove_file(base.with_extension("done"));
    std::fs::write(base.with_extension("meta"), &metadata)?;
    std::fs::write(base.with_extension("target"), format!("{name}\n{}\n{}\n", buffer.addr, buffer.len))?;
    println!("target {name}: {size} bytes at {:#x}, metadata {} bytes; waiting for the writer", buffer.addr, metadata.len());
    let started = Instant::now();
    while !base.with_extension("done").exists() {
        if started.elapsed() > Duration::from_secs(600) {
            bail!("no writer within 600 s");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // Give a late NIC write a chance to show up as such.
    std::thread::sleep(Duration::from_millis(500));
    const BLOCK: usize = 1 << 20;
    let mut expected = vec![0u8; BLOCK];
    let mut wrong = 0usize;
    let mut zero = 0usize;
    let mut listing = String::new();
    let blocks = size / BLOCK;
    for b in 0..blocks {
        let got = buffer.slice(b * BLOCK, BLOCK);
        fill_pattern(&mut expected, b * BLOCK);
        if got == expected.as_slice() {
            continue;
        }
        if got[..64].iter().all(|&x| x == 0) {
            zero += 1;
            if listing.len() < 600 {
                listing.push_str(&format!(" {b}<-zero"));
            }
            continue;
        }
        wrong += 1;
        let first = u64::from_le_bytes(got[..8].try_into().unwrap()) as usize;
        let first_word = b * BLOCK;
        let delta_mib = (first as i64 - first_word as i64) / (1 << 20);
        if listing.len() < 600 {
            listing.push_str(&format!(" {b}<-{}({delta_mib:+})", first / BLOCK));
        }
    }
    println!("VERIFY blocks={blocks} wrong={wrong} zero={zero} first:{listing}");
    let _ = std::fs::remove_file(base.with_extension("done"));
    let _ = std::fs::remove_file(base.with_extension("meta"));
    let _ = std::fs::remove_file(base.with_extension("target"));
    if wrong + zero > 0 {
        std::process::exit(2);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn writer(
    args: &Args,
    chunk: usize,
    pool: usize,
    threads: usize,
    sync: &str,
    num_workers: u32,
    rounds: usize,
    refill_delay_us: u64,
    probe: &str,
    prog_thread: bool,
) -> Result<()> {
    let base = args.dir.join(&args.id);
    let started = Instant::now();
    while !base.with_extension("target").exists() {
        if started.elapsed() > Duration::from_secs(120) {
            bail!("no target within 120 s");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(200));
    let metadata = std::fs::read(base.with_extension("meta"))?;
    let info = std::fs::read_to_string(base.with_extension("target"))?;
    let mut lines = info.lines();
    let remote_name = lines.next().context("target name")?.to_owned();
    let remote_addr: usize = lines.next().context("target addr")?.parse()?;
    let remote_len: usize = lines.next().context("target len")?.parse()?;
    let sync = match sync {
        "none" => ThreadSync::None,
        "strict" => ThreadSync::Strict,
        _ => ThreadSync::Rw,
    };
    let name = format!("reuse-writer-{}", std::process::id());
    let (agent, opt_args) = make_agent(&name, &args.devices, sync, num_workers, prog_thread)?;
    let pool_buffer = Arc::new(HostBuffer::new(pool * chunk)?);
    let _registration = agent
        .register_memory(pool_buffer.as_ref(), Some(&opt_args))
        .map_err(|e| anyhow!("register pool: {e}"))?;
    let loaded = agent.load_remote_md(&metadata).map_err(|e| anyhow!("load remote: {e}"))?;
    if loaded != remote_name {
        bail!("metadata names {loaded}, file names {remote_name}");
    }
    println!(
        "writer {name}: target {remote_name} {remote_len} bytes at {remote_addr:#x}; chunk {chunk}, pool {pool}, threads {threads}, sync {sync:?}, workers {num_workers}, rounds {rounds}"
    );
    if !probe.is_empty() {
        probe_endpoint(&agent, &pool_buffer, remote_addr, &remote_name, probe == "write")?;
    }
    let threads = threads.clamp(1, pool);
    let total_chunks = remote_len.div_ceil(chunk);
    let agent = Arc::new(agent);
    let started = Instant::now();
    let mut handles = Vec::new();
    for t in 0..threads {
        let agent = Arc::clone(&agent);
        let pool_buffer = Arc::clone(&pool_buffer);
        let remote_name = remote_name.clone();
        let my_chunks: Vec<usize> = (0..pool).filter(|c| c % threads == t).collect();
        let my_work: Vec<usize> = (0..rounds)
            .flat_map(|_| (0..total_chunks).filter(move |c| c % threads == t))
            .collect();
        handles.push(std::thread::spawn(move || -> Result<(usize, u64)> {
            let mut opt_args = OptArgs::new().map_err(|e| anyhow!("opt args: {e}"))?;
            // The backend is the agent's only one; an empty backend list lets NIXL pick it.
            let _ = &mut opt_args;
            let mut free = my_chunks;
            let mut inflight: Vec<(nixl_sys::XferRequest, usize, usize, Instant)> = Vec::new();
            let mut next = 0usize;
            let mut moved = 0u64;
            let mut polls = 0u64;
            let mut immediate = 0usize;
            let mut lat_min = f64::MAX;
            let mut lat_max = 0f64;
            let mut lat_sum = 0f64;
            let mut lat_n = 0usize;
            while next < my_work.len() || !inflight.is_empty() {
                while next < my_work.len() {
                    let Some(slot) = free.pop() else { break };
                    let work = my_work[next];
                    next += 1;
                    let offset = work * chunk;
                    let len = chunk.min(remote_len - offset);
                    fill_pattern(pool_buffer.slice_mut(slot * chunk, len), offset);
                    let local_addr = pool_buffer.addr + slot * chunk;
                    let mut local = XferDescList::new(MemType::Dram).map_err(|e| anyhow!("local list: {e}"))?;
                    local.add_desc(local_addr, len, 0);
                    let mut remote = XferDescList::new(MemType::Dram).map_err(|e| anyhow!("remote list: {e}"))?;
                    remote.add_desc(remote_addr + offset, len, 0);
                    let req = agent
                        .create_xfer_req(XferOp::Write, &local, &remote, &remote_name, None)
                        .map_err(|e| anyhow!("create: {e}"))?;
                    // The binding answers `true` while the transfer is in progress.
                    let in_progress = agent.post_xfer_req(&req, None).map_err(|e| anyhow!("post: {e}"))?;
                    if !in_progress {
                        moved += len as u64;
                        immediate += 1;
                        free.push(slot);
                    } else {
                        inflight.push((req, slot, len, Instant::now()));
                    }
                }
                let mut i = 0;
                while i < inflight.len() {
                    polls += 1;
                    match agent.get_xfer_status(&inflight[i].0).map_err(|e| anyhow!("status: {e}"))? {
                        XferStatus::Success => {
                            let (_req, slot, len, posted) = inflight.swap_remove(i);
                            let us = posted.elapsed().as_secs_f64() * 1e6;
                            lat_min = lat_min.min(us);
                            lat_max = lat_max.max(us);
                            lat_sum += us;
                            lat_n += 1;
                            moved += len as u64;
                            if refill_delay_us > 0 {
                                std::thread::sleep(Duration::from_micros(refill_delay_us));
                            }
                            free.push(slot);
                        }
                        XferStatus::InProgress => i += 1,
                    }
                }
                if inflight.len() == pool / threads && polls % 64 == 0 {
                    std::thread::yield_now();
                }
            }
            println!(
                "thread {t}: {next} transfers, {immediate} completed at post, status latency us min {:.0} avg {:.0} max {:.0}",
                if lat_n > 0 { lat_min } else { 0.0 },
                if lat_n > 0 { lat_sum / lat_n as f64 } else { 0.0 },
                lat_max
            );
            Ok((next, moved))
        }));
    }
    let mut moved = 0u64;
    for h in handles {
        let (_, m) = h.join().map_err(|_| anyhow!("thread panicked"))??;
        moved += m;
    }
    let seconds = started.elapsed().as_secs_f64();
    std::fs::write(base.with_extension("done"), "done")?;
    println!("WRITER moved={moved} seconds={seconds:.3} GBps={:.2}", moved as f64 / seconds / 1e9);
    // Keep the agent alive long enough for the target to verify before the endpoint goes away.
    std::thread::sleep(Duration::from_secs(2));
    Ok(())
}

/// One transfer, waited to completion as NIXL reports it.
fn one_transfer(agent: &Agent, op: XferOp, local_addr: usize, remote_addr: usize, len: usize, remote_name: &str) -> Result<()> {
    let mut local = XferDescList::new(MemType::Dram).map_err(|e| anyhow!("local list: {e}"))?;
    local.add_desc(local_addr, len, 0);
    let mut remote = XferDescList::new(MemType::Dram).map_err(|e| anyhow!("remote list: {e}"))?;
    remote.add_desc(remote_addr, len, 0);
    let req = agent
        .create_xfer_req(op, &local, &remote, remote_name, None)
        .map_err(|e| anyhow!("create: {e}"))?;
    if !agent.post_xfer_req(&req, None).map_err(|e| anyhow!("post: {e}"))? {
        return Ok(());
    }
    loop {
        match agent.get_xfer_status(&req).map_err(|e| anyhow!("status: {e}"))? {
            XferStatus::Success => return Ok(()),
            XferStatus::InProgress => std::thread::yield_now(),
        }
    }
}

/// Repeats small transfers on the endpoint until one provably moved data, and
/// reports how many "completed" transfers it took.
fn probe_endpoint(agent: &Agent, pool: &HostBuffer, remote_addr: usize, remote_name: &str, write: bool) -> Result<()> {
    const PAGE: usize = 4096;
    let started = Instant::now();
    let mut attempts = 0usize;
    loop {
        attempts += 1;
        let live = if write {
            // Write a pattern to the target's first page, read it back into a chunk
            // filled with something else.
            let pattern = pool.slice_mut(0, PAGE);
            pattern.fill(0xC0 | (attempts as u8 & 0x0f));
            one_transfer(agent, XferOp::Write, pool.addr, remote_addr, PAGE, remote_name)?;
            let back = pool.slice_mut(PAGE, PAGE);
            back.fill(0x11);
            one_transfer(agent, XferOp::Read, pool.addr + PAGE, remote_addr, PAGE, remote_name)?;
            pool.slice(0, PAGE) == pool.slice(PAGE, PAGE)
        } else {
            // Read the target's first page twice into chunks filled differently;
            // two real reads agree, two deferred reads leave the fillers.
            pool.slice_mut(0, PAGE).fill(0xAA);
            pool.slice_mut(PAGE, PAGE).fill(0x55);
            one_transfer(agent, XferOp::Read, pool.addr, remote_addr, PAGE, remote_name)?;
            one_transfer(agent, XferOp::Read, pool.addr + PAGE, remote_addr, PAGE, remote_name)?;
            pool.slice(0, PAGE) == pool.slice(PAGE, PAGE)
        };
        if live {
            println!("PROBE live after {attempts} attempts, {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
            return Ok(());
        }
        if started.elapsed() > Duration::from_secs(10) {
            bail!("endpoint never moved data in 10 s ({attempts} attempts)");
        }
        std::thread::sleep(Duration::from_micros(200));
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match &args.role {
        Role::Target { size } => target(&args, *size),
        Role::Writer { chunk, pool, threads, sync, num_workers, rounds, refill_delay_us, probe, prog_thread } => {
            writer(&args, *chunk, *pool, *threads, sync, *num_workers, *rounds, *refill_delay_us, probe, *prog_thread)
        }
    }
}
