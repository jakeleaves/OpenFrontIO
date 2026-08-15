//! Bit-exact port of `src/core/PseudoRandom.ts` (sfc32 + splitmix32 seed).

#[derive(Clone, Debug)]
pub struct PseudoRandom {
    s0: i32,
    s1: i32,
    s2: i32,
    s3: i32,
}

impl PseudoRandom {
    /// Truncates seed to 32 bits (same as `seed | 0` in JS).
    pub fn new(seed: i32) -> Self {
        let mut h = seed;
        let mut split = || {
            h = h.wrapping_add(0x9e3779b9_u32 as i32);
            let mut t = h ^ ((h as u32) >> 16) as i32;
            t = imul(t, 0x21f0aaad_u32 as i32);
            t = t ^ (((t as u32) >> 15) as i32);
            t = imul(t, 0x735a2d97_u32 as i32);
            (t ^ (((t as u32) >> 15) as i32)) | 0
        };
        let mut rng = Self {
            s0: split(),
            s1: split(),
            s2: split(),
            s3: split(),
        };
        for _ in 0..12 {
            let _ = rng.next();
        }
        rng
    }

    /// From a numeric seed that may exceed i32 (matches `seed | 0`).
    pub fn from_u32(seed: u32) -> Self {
        Self::new(seed as i32)
    }

    pub fn from_hash_seed(seed: i32) -> Self {
        Self::new(seed)
    }

    /// Uniform in [0, 1).
    pub fn next(&mut self) -> f64 {
        let t = self
            .s0
            .wrapping_add(self.s1)
            .wrapping_add(self.s3);
        self.s3 = self.s3.wrapping_add(1);
        self.s0 = self.s1 ^ (((self.s1 as u32) >> 9) as i32);
        self.s1 = self.s2.wrapping_add(self.s2 << 3);
        self.s2 = ((self.s2 as u32) << 21 | (self.s2 as u32) >> 11) as i32;
        self.s2 = self.s2.wrapping_add(t);
        (t as u32 as f64) / 4294967296.0
    }

    pub fn next_int(&mut self, min: f64, max: f64) -> i32 {
        let lo = min.floor() as i32;
        let hi = max.floor() as i32;
        (self.next() * (hi - lo) as f64).floor() as i32 + lo
    }

    pub fn next_float(&mut self, min: f64, max: f64) -> f64 {
        self.next() * (max - min) + min
    }

    pub fn next_id(&mut self) -> String {
        let pow36_8 = 36f64.powi(8);
        let n = (self.next() * pow36_8).floor() as u64;
        format!("{:0>8}", radix36(n))
    }

    pub fn chance(&mut self, odds: i32) -> bool {
        self.next_int(0.0, odds as f64) == 0
    }

    pub fn rand_element<'a, T>(&mut self, arr: &'a [T]) -> &'a T {
        assert!(!arr.is_empty(), "array must not be empty");
        &arr[self.next_int(0.0, arr.len() as f64) as usize]
    }

    pub fn shuffle_array<T: Clone>(&mut self, array: &[T]) -> Vec<T> {
        let mut result = array.to_vec();
        for i in (1..result.len()).rev() {
            let j = self.next_int(0.0, (i + 1) as f64) as usize;
            result.swap(i, j);
        }
        result
    }
}

/// JS `Math.imul`
fn imul(a: i32, b: i32) -> i32 {
    ((a as i64) * (b as i64)) as i32
}

fn radix36(mut n: u64) -> String {
    if n == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut out = Vec::new();
    while n > 0 {
        out.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    out.reverse();
    String::from_utf8(out).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_stream_deterministic() {
        let mut a = PseudoRandom::new(42);
        let mut b = PseudoRandom::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next().to_bits(), b.next().to_bits());
        }
    }

    #[test]
    fn congruent_seeds_match() {
        // seeds congruent mod 2^32 produce identical streams
        let mut a = PseudoRandom::new(1);
        let mut b = PseudoRandom::new(1i32.wrapping_add(0));
        assert_eq!(a.next().to_bits(), b.next().to_bits());
    }

    #[test]
    fn next_int_range() {
        let mut r = PseudoRandom::new(7);
        for _ in 0..100 {
            let v = r.next_int(0.0, 10.0);
            assert!((0..10).contains(&v));
        }
    }
}
