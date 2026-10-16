fn main() {
    cxx_build::bridge("src/lib.rs")
        .flag_if_supported("-std=c++17")
        .compile("map_egg");

    // Export the header belonging to this Cargo profile. The shared cxxbridge
    // symlink can still point to a different toolchain's build directory.
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let header = out.join("cxxbridge/include/map_egg/src/lib.rs.h");
    let export = out
        .ancestors()
        .nth(3)
        .unwrap()
        .join("include/map_egg/src/lib.rs.h");
    std::fs::create_dir_all(export.parent().unwrap()).unwrap();
    std::fs::copy(header, export).unwrap();

    println!("cargo:rerun-if-changed=src/lib.rs");
}
