use std::process::Command;
use std::io::prelude::*;

macro_rules! mf_dir {
    ($p:literal) => {
        concat!(env!("CARGO_MANIFEST_DIR"), $p)
    };
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let build_output = Command::new(mf_dir!("/src/build_model.py"))
        .arg(&out_dir)
        .output()
        .expect("Failed to build model");
    assert!(
        ["model.o", "graph.json", "params.bin"]
            .iter()
            .all(|f| { std::path::Path::new(&format!("{}/{}", out_dir, f)).exists() }),
        "Could not build tvm lib: STDOUT:\n\n{}\n\nSTDERR\n\n{}",
        String::from_utf8(build_output.stdout).unwrap().trim(),
        String::from_utf8(build_output.stderr).unwrap().trim()
    );

    let sysroot_output = Command::new("rustc")
        .args(&["--print", "sysroot"])
        .output()
        .expect("Failed to get sysroot");
    let sysroot = String::from_utf8(sysroot_output.stdout).unwrap();
    let sysroot = sysroot.trim();
    let mut llvm_tools_path = std::path::PathBuf::from(&sysroot);
    llvm_tools_path.push("lib/rustlib/x86_64-unknown-linux-gnu/bin");

    Command::new("rustup")
        .args(&["component", "add", "llvm-tools-preview"])
        .output()
        .expect("failed to install llvm tools");

    std::process::Command::new(llvm_tools_path.join("llvm-objcopy"))
        .arg("--globalize-symbol=__tvm_module_startup")
        .arg("--remove-section=.ctors")
        .arg(&format!("{}/model.o", out_dir))
        .output()
        .expect("gould not gloablize startup function");

    std::process::Command::new(llvm_tools_path.join("llvm-ar"))
        .arg("rcs")
        .arg(&format!("{}/libmodel.a", out_dir))
        .arg(&format!("{}/model.o", out_dir))
        .output()
        .expect("failed to package model archive");

    println!("cargo:rustc-link-lib=static=model");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
