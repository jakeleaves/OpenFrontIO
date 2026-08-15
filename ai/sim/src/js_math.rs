//! JS Math helpers that must match V8 / Node for hash parity.
//!
//! Rust libm and JS Math can disagree on pow/exp edge cases. Prefer these
//! wrappers for any formula copied from `Config.ts` / `Util.ts`.

/// JS `Math.min(Math.max(value, min), max)`
#[inline]
pub fn within(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Port of `Util.sigmoid`
#[inline]
pub fn sigmoid(value: f64, decay_rate: f64, midpoint: f64) -> f64 {
    1.0 / (1.0 + (-decay_rate * (value - midpoint)).exp())
}

/// JS `Math.pow` — use for economy/combat so we can swap to a JS-compatible
/// softfloat later if parity diffs appear.
#[inline]
pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// JS `Math.floor` for non-negative values used in gold/troop flooring.
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// JS `Math.abs` then used by simpleHash (abs of i32).
#[inline]
pub fn abs_i32(x: i32) -> i32 {
    x.wrapping_abs()
}

/// Port of `Util.simpleHash` — Java-style string hash, then Math.abs.
pub fn simple_hash(s: &str) -> i32 {
    let mut hash: i32 = 0;
    for b in s.chars() {
        let char_code = b as u32 as i32;
        hash = hash.wrapping_shl(5).wrapping_sub(hash).wrapping_add(char_code);
        hash &= hash; // no-op in Rust but mirrors TS `hash = hash & hash`
    }
    hash.wrapping_abs()
}

/// Port of `Util.toInt` — floor to bigint (we use i64 for troops in some paths;
/// gold stays i64/u64).
#[inline]
pub fn to_int_floor(num: f64) -> i64 {
    if num.is_infinite() {
        if num.is_sign_positive() {
            return i64::MAX;
        }
        return i64::MIN;
    }
    num.floor() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_hash_known() {
        // Cross-checked against Node `Util.simpleHash`
        assert_eq!(simple_hash("test"), 3556498);
        assert_eq!(simple_hash("player1"), 493567632);
        assert_eq!(simple_hash("abc"), 96354);
        assert_eq!(simple_hash("game-123"), 1770375721);
        assert_eq!(simple_hash("human0"), 1206139741);
    }

    #[test]
    fn within_clamps() {
        assert_eq!(within(5.0, 0.0, 3.0), 3.0);
        assert_eq!(within(-1.0, 0.0, 3.0), 0.0);
        assert_eq!(within(1.5, 0.0, 3.0), 1.5);
    }
}
