//! Combinatorial equivalence checking (CEC) via ABC: golden AIG vs mapped Verilog.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

static ABC_FOR_CEC: OnceLock<Option<PathBuf>> = OnceLock::new();

fn abc_for_cec() -> Option<PathBuf> {
    ABC_FOR_CEC
        .get_or_init(|| {
            if let Ok(p) = std::env::var("BENCH_RS_EMAP_ABC") {
                let pb = PathBuf::from(p.trim());
                return if pb.as_os_str().is_empty() || !pb.is_file() {
                    None
                } else {
                    Some(pb)
                };
            }
            ["yosys-abc", "abc"]
                .into_iter()
                .find(|n| Command::new(n).arg("-h").output().is_ok())
                .map(PathBuf::from)
        })
        .clone()
}

fn parse_cec_outcome(txt: &str) -> Option<f64> {
    let lower = txt.to_ascii_lowercase();
    if lower.contains("networks are equivalent") || txt.contains("UNSAT") {
        return Some(1.0);
    }
    if lower.contains("networks are not equivalent") {
        return Some(0.0);
    }
    if txt.contains("SAT") && !txt.contains("UNSAT") {
        return Some(0.0);
    }
    None
}

fn shell_path(p: &Path) -> String {
    let s = p.to_string_lossy();
    if s.contains(' ') {
        format!("\"{s}\"")
    } else {
        s.into_owned()
    }
}

fn abs(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

fn cec_debug(msg: &str) {
    if std::env::var("BENCH_RS_EMAP_CEC_DEBUG").as_deref() == Ok("1") {
        eprintln!("{msg}");
    }
}

fn nanos_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos().to_string())
        .unwrap_or_else(|_| "0".into())
}

/// Run ABC `cec -n` between a golden AIG and a netlist built from `mapped_verilog` + `genlib`.
///
/// Returns `1.0` if equivalent, `0.0` if not, `-1.0` if ABC is missing, the Verilog→AIG build
/// failed, or the CEC output could not be parsed.
///
/// Environment:
/// - `BENCH_RS_EMAP_ABC`: path to `yosys-abc` or `abc` executable (optional; otherwise PATH).
/// - `BENCH_RS_EMAP_CEC_DEBUG=1`: print diagnostics on failure.
pub fn run_abc_cec(genlib: &Path, aig: &Path, mapped_verilog: &Path) -> f64 {
    let Some(exe) = abc_for_cec() else {
        cec_debug("BENCH_RS_EMAP_CEC_DEBUG: no ABC (set BENCH_RS_EMAP_ABC or PATH yosys-abc/abc)");
        return -1.0;
    };
    let g = abs(genlib);
    let a = abs(aig);
    let v = abs(mapped_verilog);
    let mut impl_aig = std::env::temp_dir();
    impl_aig.push(format!(
        "emapper_cec_impl_{}_{}.aig",
        std::process::id(),
        nanos_stamp()
    ));
    let impl_s = shell_path(&impl_aig);

    let build = format!(
        "read_library {}; read_verilog -m {}; strash; write_aiger {}",
        shell_path(&g),
        shell_path(&v),
        impl_s
    );
    let out_b = match Command::new(&exe).args(["-c", &build]).output() {
        Ok(o) => o,
        Err(_) => {
            cec_debug("BENCH_RS_EMAP_CEC_DEBUG: spawn failed (read_library/read_verilog)");
            return -1.0;
        }
    };
    if !out_b.status.success() || !impl_aig.exists() {
        cec_debug(&format!(
            "BENCH_RS_EMAP_CEC_DEBUG: ABC build failed (exit {:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
            out_b.status.code(),
            String::from_utf8_lossy(&out_b.stdout),
            String::from_utf8_lossy(&out_b.stderr)
        ));
        let _ = std::fs::remove_file(&impl_aig);
        return -1.0;
    }

    let cec = format!("cec -n {} {}", shell_path(&a), impl_s);
    let out = match Command::new(&exe).args(["-c", &cec]).output() {
        Ok(o) => o,
        Err(_) => {
            cec_debug("BENCH_RS_EMAP_CEC_DEBUG: spawn failed (cec)");
            let _ = std::fs::remove_file(&impl_aig);
            return -1.0;
        }
    };
    let _ = std::fs::remove_file(&impl_aig);

    let txt = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if let Some(x) = parse_cec_outcome(&txt) {
        return x;
    }
    cec_debug(&format!(
        "BENCH_RS_EMAP_CEC_DEBUG: unrecognized cec output (exit {:?})\n{txt}",
        out.status.code()
    ));
    -1.0
}
