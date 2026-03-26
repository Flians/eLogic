//! Run Berkeley ABC / yosys-abc `print_stats` on a mapped Verilog netlist so QoR
//! matches interactive `read_genlib` + `read -m` + `print_stats`.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug)]
pub struct AbcPrintStats {
    pub nd: usize,
    pub area: f32,
    pub delay: f32,
    pub lev: usize,
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(x) = chars.next() {
                    if (0x40..=0x7E).contains(&(x as u32)) {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

fn parse_key_usize(line: &str, key: &str) -> Option<usize> {
    let needle = format!("{key} =");
    let i = line.find(&needle)?;
    let rest = line[i + needle.len()..].trim_start();
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num.parse().ok()
}

fn parse_key_f32(line: &str, key: &str) -> Option<f32> {
    let needle = format!("{key} =");
    let i = line.find(&needle)?;
    let rest = line[i + needle.len()..].trim_start();
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    num.parse().ok()
}

fn parse_print_stats(stdout: &str) -> Option<AbcPrintStats> {
    let text = strip_ansi(stdout);
    let line = text
        .lines()
        .find(|l| l.contains("nd") && l.contains("area") && l.contains("delay"))?;
    Some(AbcPrintStats {
        nd: parse_key_usize(line, "nd")?,
        area: parse_key_f32(line, "area")?,
        delay: parse_key_f32(line, "delay")?,
        lev: parse_key_usize(line, "lev")?,
    })
}

fn shell_escape_path(p: &Path) -> String {
    let s = p.to_string_lossy();
    if s.contains(' ') || s.contains('\'') {
        format!("\"{s}\"")
    } else {
        s.into_owned()
    }
}

/// Prefer `BENCH_RS_EMAP_ABC`, then `yosys-abc`, then `abc` on `PATH`.
pub fn resolve_abc_executable() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BENCH_RS_EMAP_ABC") {
        let pb = PathBuf::from(p.trim());
        if pb.as_os_str().is_empty() {
            return None;
        }
        if pb.is_file() {
            return Some(pb);
        }
    }
    for name in ["yosys-abc", "abc"] {
        if Command::new(name).arg("-h").output().is_ok() {
            return Some(PathBuf::from(name));
        }
    }
    None
}

/// `read_genlib` + `read -m` + `print_stats` (same flow as typical ABC QoR readout).
pub fn run_abc_print_stats(genlib: &Path, verilog: &Path) -> Option<AbcPrintStats> {
    let exe = resolve_abc_executable()?;
    let abs_g = std::fs::canonicalize(genlib).ok()?;
    let abs_v = std::fs::canonicalize(verilog).ok()?;
    let g = shell_escape_path(&abs_g);
    let v = shell_escape_path(&abs_v);
    let script = format!("read_genlib {g}; read -m {v}; print_stats");
    let out = Command::new(&exe).args(["-c", &script]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    parse_print_stats(&txt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_print_stats_matches_abc_line() {
        let s = "\x1b[1;37mtop                           :\x1b[0m i/o =  256/  129  lat =    0  nd =   899  edge =   1662  area =92.60  delay =2843.97  lev = 129";
        let p = parse_print_stats(s).expect("parse");
        assert_eq!(p.nd, 899);
        assert!((p.area - 92.60).abs() < 0.01);
        assert!((p.delay - 2843.97).abs() < 0.01);
        assert_eq!(p.lev, 129);
    }
}
