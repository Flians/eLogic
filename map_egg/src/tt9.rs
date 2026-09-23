//! Truth tables up to **9 inputs** (512 rows) as eight `u64` limbs (little-endian bit order).

pub type Tt9 = [u64; 8];

#[inline]
pub fn tt9_zero() -> Tt9 {
    [0u64; 8]
}

#[inline]
pub fn tt9_bit_get(tt: &Tt9, bit: usize) -> bool {
    debug_assert!(bit < 512);
    let w = bit / 64;
    let b = bit % 64;
    ((tt[w] >> b) & 1) == 1
}

#[inline]
pub fn tt9_bit_set(tt: &mut Tt9, bit: usize) {
    let w = bit / 64;
    let b = bit % 64;
    tt[w] |= 1u64 << b;
}

#[must_use]
pub fn tt9_from_u64(lo: u64) -> Tt9 {
    let mut t = tt9_zero();
    t[0] = lo;
    t
}

/// All minterms true for `n` inputs (lower `2^n` bits).
#[must_use]
pub fn tt9_all_ones(n: u8) -> Tt9 {
    let mut t = tt9_zero();
    for i in 0..(1usize << usize::from(n)) {
        tt9_bit_set(&mut t, i);
    }
    t
}

#[must_use]
pub fn tt9_and_masked(a: &Tt9, b: &Tt9, n: u8) -> Tt9 {
    let mut o = tt9_zero();
    for i in 0..(1usize << usize::from(n)) {
        if tt9_bit_get(a, i) && tt9_bit_get(b, i) {
            tt9_bit_set(&mut o, i);
        }
    }
    o
}

/// Invert only the lower `2^n` bits (valid assignments); clears bits `>= 2^n`.
#[must_use]
pub fn tt9_not_masked(tt: &Tt9, n: u8) -> Tt9 {
    let mut out = tt9_zero();
    let bits = 1usize << usize::from(n);
    for i in 0..bits.min(512) {
        if !tt9_bit_get(tt, i) {
            tt9_bit_set(&mut out, i);
        }
    }
    out
}

#[must_use]
pub fn permute_negate_tt9(tt: &Tt9, n: u8, perm: &[u8], neg_mask: u16, out_neg: bool) -> Tt9 {
    let n = n as usize;
    let mut out = tt9_zero();
    for a in 0..(1usize << n) {
        let mut old = 0usize;
        for i in 0..n {
            let mut bit = ((a >> i) & 1) == 1;
            if ((neg_mask >> i) & 1) == 1 {
                bit = !bit;
            }
            if bit {
                old |= 1usize << usize::from(perm[i]);
            }
        }
        let mut y = tt9_bit_get(tt, old);
        if out_neg {
            y = !y;
        }
        if y {
            tt9_bit_set(&mut out, a);
        }
    }
    out
}

/// Fixed-width hex for egg `gate:` symbols (8×16 hex digits = 512 bits).
#[must_use]
pub fn tt9_to_hex(tt: &Tt9) -> String {
    tt.iter().map(|w| format!("{w:016x}")).collect()
}

#[must_use]
pub fn tt9_from_hex(s: &str) -> Option<Tt9> {
    if s.len() != 128 {
        return None;
    }
    let mut out = tt9_zero();
    for (i, chunk) in s.as_bytes().chunks(16).enumerate().take(8) {
        let hex = std::str::from_utf8(chunk).ok()?;
        out[i] = u64::from_str_radix(hex, 16).ok()?;
    }
    Some(out)
}
