//! Multi-objective optimization utilities.
//!
//! Provides Pareto dominance checking, non-dominated sorting, and crowding
//! distance computation for multi-objective optimization.

use crate::study::Direction;

/// Check if solution `a` dominates solution `b`.
///
/// `a` dominates `b` if `a` is at least as good in all objectives and strictly
/// better in at least one.
pub fn dominates(a: &[f64], b: &[f64], directions: &[Direction]) -> bool {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), directions.len());

    let mut dominated_in_any = false;
    for i in 0..a.len() {
        let cmp = match directions[i] {
            Direction::Minimize => a[i].partial_cmp(&b[i]),
            Direction::Maximize => b[i].partial_cmp(&a[i]),
        };
        match cmp {
            Some(std::cmp::Ordering::Greater) => return false, // a is worse in this objective
            Some(std::cmp::Ordering::Less) => dominated_in_any = true, // a is better
            _ => {}                                                     // equal or NaN
        }
    }
    dominated_in_any
}

/// Perform non-dominated sorting on a set of objective value vectors.
///
/// Returns a vector of fronts, where each front is a vector of indices into
/// the input `values` slice. Front 0 is the Pareto front.
pub fn non_dominated_sort(
    values: &[Vec<f64>],
    directions: &[Direction],
) -> Vec<Vec<usize>> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }

    // For each solution, count how many solutions dominate it and which solutions it dominates
    let mut domination_count = vec![0usize; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&values[i], &values[j], directions) {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if dominates(&values[j], &values[i], directions) {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts = vec![];
    let mut current_front: Vec<usize> = (0..n)
        .filter(|&i| domination_count[i] == 0)
        .collect();

    while !current_front.is_empty() {
        let mut next_front = vec![];
        for &i in &current_front {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Compute crowding distances for solutions within a single front.
///
/// Returns a vector of crowding distances (one per solution in the front).
/// Boundary solutions get `f64::INFINITY`.
pub fn crowding_distance(values: &[Vec<f64>], directions: &[Direction]) -> Vec<f64> {
    let n = values.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let n_objectives = directions.len();
    let mut distances = vec![0.0_f64; n];

    for m in 0..n_objectives {
        // Sort indices by objective m
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            values[a][m]
                .partial_cmp(&values[b][m])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary solutions get infinity
        distances[indices[0]] = f64::INFINITY;
        distances[indices[n - 1]] = f64::INFINITY;

        let range = values[indices[n - 1]][m] - values[indices[0]][m];
        if range <= 0.0 {
            continue;
        }

        for i in 1..(n - 1) {
            distances[indices[i]] +=
                (values[indices[i + 1]][m] - values[indices[i - 1]][m]) / range;
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates_minimize() {
        let dirs = vec![Direction::Minimize, Direction::Minimize];
        // (1, 2) dominates (2, 3)
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0], &dirs));
        // (2, 3) does not dominate (1, 2)
        assert!(!dominates(&[2.0, 3.0], &[1.0, 2.0], &dirs));
        // Neither dominates (tradeoff)
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0], &dirs));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0], &dirs));
    }

    #[test]
    fn test_dominates_maximize() {
        let dirs = vec![Direction::Maximize, Direction::Maximize];
        // (3, 4) dominates (2, 3)
        assert!(dominates(&[3.0, 4.0], &[2.0, 3.0], &dirs));
        assert!(!dominates(&[2.0, 3.0], &[3.0, 4.0], &dirs));
    }

    #[test]
    fn test_dominates_mixed() {
        let dirs = vec![Direction::Minimize, Direction::Maximize];
        // (1, 4) dominates (2, 3): better in both (lower first, higher second)
        assert!(dominates(&[1.0, 4.0], &[2.0, 3.0], &dirs));
        assert!(!dominates(&[2.0, 3.0], &[1.0, 4.0], &dirs));
    }

    #[test]
    fn test_non_dominated_sort_simple() {
        let dirs = vec![Direction::Minimize, Direction::Minimize];
        let values = vec![
            vec![1.0, 4.0], // Pareto front
            vec![2.0, 3.0], // Pareto front
            vec![3.0, 2.0], // Pareto front
            vec![2.0, 4.0], // Dominated by 0 and 1
            vec![3.0, 3.0], // Dominated by 1 and 2
        ];

        let fronts = non_dominated_sort(&values, &dirs);
        assert_eq!(fronts.len(), 2);

        // Front 0 should contain indices 0, 1, 2
        let mut front0 = fronts[0].clone();
        front0.sort();
        assert_eq!(front0, vec![0, 1, 2]);

        // Front 1 should contain indices 3, 4
        let mut front1 = fronts[1].clone();
        front1.sort();
        assert_eq!(front1, vec![3, 4]);
    }

    #[test]
    fn test_non_dominated_sort_empty() {
        let dirs = vec![Direction::Minimize];
        assert!(non_dominated_sort(&[], &dirs).is_empty());
    }

    #[test]
    fn test_non_dominated_sort_single() {
        let dirs = vec![Direction::Minimize, Direction::Minimize];
        let values = vec![vec![1.0, 2.0]];
        let fronts = non_dominated_sort(&values, &dirs);
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0], vec![0]);
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let dirs = vec![Direction::Minimize, Direction::Minimize];
        let values = vec![vec![1.0, 4.0], vec![3.0, 2.0]];
        let distances = crowding_distance(&values, &dirs);
        assert!(distances[0].is_infinite());
        assert!(distances[1].is_infinite());
    }

    #[test]
    fn test_crowding_distance_three_points() {
        let dirs = vec![Direction::Minimize, Direction::Minimize];
        let values = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
        ];
        let distances = crowding_distance(&values, &dirs);
        assert!(distances[0].is_infinite());
        assert!(distances[2].is_infinite());
        // Middle point should have finite distance
        assert!(distances[1].is_finite());
        assert!(distances[1] > 0.0);
    }
}
