//! Smoke test: print the system Metal device name and free bytes.
//!
//! Run with: `cargo run --example metal_smoke --features metal -p ltx-budget`

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use ltx_budget::{DeviceMemory, MetalDevice};

        match MetalDevice::system_default() {
            Ok(dev) => {
                let free = dev.free_bytes().unwrap_or(u64::MAX);
                let total = dev.total_bytes().unwrap_or(u64::MAX);
                println!("Metal device : {}", dev.device_name());
                println!("Total memory : {} MiB", total / (1024 * 1024));
                println!("Free memory  : {} MiB", free / (1024 * 1024));
            }
            Err(e) => {
                eprintln!("Metal device query failed: {e}");
                std::process::exit(1);
            }
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        eprintln!("Run with --features metal on macOS");
    }
}
