//! Water BFS pathfinding (M6 foundation; HPA* can replace later).

use crate::map::{GameMap, TileRef};
use std::collections::VecDeque;

pub fn bfs_water(map: &GameMap, start: TileRef, goal: TileRef) -> Option<Vec<TileRef>> {
    if start == goal {
        return Some(vec![start]);
    }
    if !map.is_water(start) || !map.is_water(goal) {
        return None;
    }
    let n = map.len();
    let mut prev: Vec<Option<TileRef>> = vec![None; n];
    let mut seen = vec![false; n];
    let mut q = VecDeque::new();
    seen[start as usize] = true;
    q.push_back(start);
    while let Some(cur) = q.pop_front() {
        if cur == goal {
            let mut path = Vec::new();
            let mut c = goal;
            path.push(c);
            while c != start {
                c = prev[c as usize]?;
                path.push(c);
            }
            path.reverse();
            return Some(path);
        }
        let mut nbuf = [0u32; 4];
        let cnt = map.neighbors4(cur, &mut nbuf);
        for i in 0..cnt {
            let nb = nbuf[i];
            if map.is_water(nb) && !seen[nb as usize] {
                seen[nb as usize] = true;
                prev[nb as usize] = Some(cur);
                q.push_back(nb);
            }
        }
    }
    None
}
