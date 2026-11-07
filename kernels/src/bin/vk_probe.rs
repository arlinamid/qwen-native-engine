//! Finds the step at which a wgpu backend crashes, printing before each one.
//!
//! cargo run --release -p qwen-kernels --bin vk_probe -- vulkan|dx12 [f16]

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use std::io::Write;
    let step = |s: &str| {
        eprintln!("step: {s}");
        let _ = std::io::stderr().flush();
    };
    let args: Vec<String> = std::env::args().collect();
    let backends = match args.get(1).map(String::as_str) {
        Some("dx12") => wgpu::Backends::DX12,
        _ => wgpu::Backends::VULKAN,
    };
    let want_f16 = args.iter().any(|a| a == "f16");

    step("create instance");
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    if let Some(path) = args.iter().find_map(|a| a.strip_prefix("dxc=")) {
        desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc { dxc_path: path.to_string() };
    }
    let instance = wgpu::Instance::new(desc);

    step("enumerate adapters");
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    for a in &adapters {
        let i = a.get_info();
        eprintln!("  {} | {:?} | driver '{}' '{}' | f16 {}", i.name, i.backend, i.driver, i.driver_info, a.features().contains(wgpu::Features::SHADER_F16));
    }

    step("request adapter");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("adapter");

    step("request device (default limits)");
    let required_features = if want_f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() };
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features, ..Default::default() })).expect("device");
    drop(device);

    step("request device (adapter limits)");
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");

    step("compile a compute pipeline");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> y: array<f32>; @compute @workgroup_size(64) fn main() { y[0] = 1.0; }".into()),
    });
    let _p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    step("done");
}
