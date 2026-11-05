//! The three CUDA runtime calls the benchmark needs, loaded at run time so
//! the binary builds and runs on hosts without the CUDA toolkit.

use anyhow::{anyhow, Context, Result};
use libloading::{Library, Symbol};
use std::ffi::c_void;

pub struct Cuda {
    lib: Library,
}

impl Cuda {
    pub fn load() -> Result<Self> {
        // SAFETY: loading the CUDA runtime library runs its constructors only.
        let lib =
            unsafe { Library::new("libcudart.so").or_else(|_| Library::new("libcudart.so.13")) }
                .context("libcudart.so not found (set LD_LIBRARY_PATH to the CUDA runtime)")?;
        Ok(Self { lib })
    }

    fn check(&self, what: &str, rc: i32) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(anyhow!("{what} failed with CUDA error {rc}"))
        }
    }

    pub fn set_device(&self, device: i32) -> Result<()> {
        // SAFETY: the symbol has this C signature in every CUDA runtime.
        let f: Symbol<unsafe extern "C" fn(i32) -> i32> =
            unsafe { self.lib.get(b"cudaSetDevice\0")? };
        self.check("cudaSetDevice", unsafe { f(device) })
    }

    pub fn malloc(&self, len: usize) -> Result<usize> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: as above; `ptr` is a valid out-pointer.
        let f: Symbol<unsafe extern "C" fn(*mut *mut c_void, usize) -> i32> =
            unsafe { self.lib.get(b"cudaMalloc\0")? };
        self.check("cudaMalloc", unsafe { f(&mut ptr, len) })?;
        Ok(ptr as usize)
    }

    pub fn memset(&self, ptr: usize, value: i32, len: usize) -> Result<()> {
        // SAFETY: as above; the range was allocated by `malloc`.
        let f: Symbol<unsafe extern "C" fn(*mut c_void, i32, usize) -> i32> =
            unsafe { self.lib.get(b"cudaMemset\0")? };
        self.check("cudaMemset", unsafe { f(ptr as *mut c_void, value, len) })?;
        self.synchronize()
    }

    pub fn copy_to_host(&self, dst: &mut [u8], src: usize) -> Result<()> {
        const DEVICE_TO_HOST: i32 = 2;
        // SAFETY: as above; `dst` is writable for its length and `src` holds at least that many bytes.
        let f: Symbol<unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32> =
            unsafe { self.lib.get(b"cudaMemcpy\0")? };
        self.check("cudaMemcpy", unsafe {
            f(
                dst.as_mut_ptr() as *mut c_void,
                src as *const c_void,
                dst.len(),
                DEVICE_TO_HOST,
            )
        })
    }

    /// Copies `src` to device memory at `dst`.
    pub fn copy_from_host(&self, dst: usize, src: &[u8]) -> Result<()> {
        const HOST_TO_DEVICE: i32 = 1;
        // SAFETY: as above; `dst` holds at least `src.len()` writable bytes.
        let f: Symbol<unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32> =
            unsafe { self.lib.get(b"cudaMemcpy\0")? };
        self.check("cudaMemcpy", unsafe {
            f(
                dst as *mut c_void,
                src.as_ptr() as *const c_void,
                src.len(),
                HOST_TO_DEVICE,
            )
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: as above.
        let f: Symbol<unsafe extern "C" fn() -> i32> =
            unsafe { self.lib.get(b"cudaDeviceSynchronize\0")? };
        self.check("cudaDeviceSynchronize", unsafe { f() })
    }
}
