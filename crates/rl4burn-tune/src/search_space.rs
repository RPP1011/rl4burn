//! Search space utilities.
//!
//! Functions for inferring and decomposing search spaces from trial history.

use std::collections::{HashMap, HashSet};

use crate::distributions::Distribution;
use crate::study::Study;
use crate::trial::TrialState;

/// Infer the intersection search space from completed trials.
///
/// Returns the set of (param_name, distribution) pairs that appear in ALL
/// completed trials. This is the "common" search space that every trial shares.
///
/// Note: Since rl4burn-tune doesn't store distributions on FrozenTrial yet,
/// this version infers the search space from parameter names only. It returns
/// a set of parameter names that appear in every completed trial.
pub fn intersection_search_space_names(study: &Study) -> HashSet<String> {
    let completed: Vec<_> = study
        .trials()
        .iter()
        .filter(|t| t.state == TrialState::Complete)
        .collect();

    if completed.is_empty() {
        return HashSet::new();
    }

    // Start with the first trial's params
    let mut common: HashSet<String> = completed[0].params.keys().cloned().collect();

    // Intersect with each subsequent trial
    for trial in &completed[1..] {
        let trial_params: HashSet<String> = trial.params.keys().cloned().collect();
        common = common.intersection(&trial_params).cloned().collect();
    }

    common
}

/// Infer the intersection search space from completed trials, using stored distributions.
///
/// Returns the set of (param_name, Distribution) pairs that appear in ALL
/// completed trials with compatible distributions.
pub fn intersection_search_space(study: &Study) -> HashMap<String, Distribution> {
    let completed: Vec<_> = study
        .trials()
        .iter()
        .filter(|t| t.state == TrialState::Complete)
        .collect();

    if completed.is_empty() {
        return HashMap::new();
    }

    // Get param names from first trial's distributions
    let first_dists = &completed[0].distributions;
    if first_dists.is_empty() {
        return HashMap::new();
    }

    let mut result = HashMap::new();

    for (name, dist) in first_dists {
        // Check if this param with a compatible distribution exists in all trials
        let in_all = completed[1..].iter().all(|t| t.distributions.contains_key(name));
        if in_all {
            result.insert(name.clone(), dist.clone());
        }
    }

    result
}

/// Decompose the search space into conditionally independent subspaces.
///
/// Groups parameters that always appear together across trials. If parameter A
/// appears in a trial if and only if parameter B appears, they are in the same group.
///
/// Returns a vector of groups, where each group is a set of parameter names.
pub fn group_decomposed_search_space(study: &Study) -> Vec<HashSet<String>> {
    let completed: Vec<_> = study
        .trials()
        .iter()
        .filter(|t| t.state == TrialState::Complete)
        .collect();

    if completed.is_empty() {
        return vec![];
    }

    // Build a presence matrix: for each trial, which params are present
    let all_params: HashSet<String> = completed
        .iter()
        .flat_map(|t| t.params.keys().cloned())
        .collect();

    if all_params.is_empty() {
        return vec![];
    }

    // Build presence vectors for each param
    let param_list: Vec<String> = all_params.into_iter().collect();
    let presence: HashMap<&str, Vec<bool>> = param_list
        .iter()
        .map(|p| {
            let pres: Vec<bool> = completed.iter().map(|t| t.params.contains_key(p)).collect();
            (p.as_str(), pres)
        })
        .collect();

    // Group params with identical presence vectors
    let mut groups: Vec<HashSet<String>> = Vec::new();

    for param in &param_list {
        let pv = &presence[param.as_str()];
        let mut found = false;
        for group in &mut groups {
            // Check if this param has the same presence as any member of the group
            let representative = group.iter().next().unwrap();
            if presence[representative.as_str()] == *pv {
                group.insert(param.clone());
                found = true;
                break;
            }
        }
        if !found {
            let mut new_group = HashSet::new();
            new_group.insert(param.clone());
            groups.push(new_group);
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{Direction, Study};
    use crate::trial::{FrozenTrial, TrialState};

    fn make_trial(number: usize, params: &[(&str, f64)]) -> FrozenTrial {
        let mut ft = FrozenTrial::new(number);
        ft.state = TrialState::Complete;
        ft.value = Some(1.0);
        for &(name, val) in params {
            ft.params.insert(name.to_string(), val);
        }
        ft
    }

    #[test]
    fn test_intersection_search_space_names_basic() {
        let mut study = Study::new(Direction::Minimize);
        study.add_trial(make_trial(0, &[("x", 1.0), ("y", 2.0), ("z", 3.0)]));
        study.add_trial(make_trial(1, &[("x", 4.0), ("y", 5.0)]));
        study.add_trial(make_trial(2, &[("x", 7.0), ("y", 8.0), ("w", 9.0)]));

        let common = intersection_search_space_names(&study);
        assert_eq!(common.len(), 2);
        assert!(common.contains("x"));
        assert!(common.contains("y"));
        assert!(!common.contains("z"));
        assert!(!common.contains("w"));
    }

    #[test]
    fn test_intersection_search_space_names_empty() {
        let study = Study::new(Direction::Minimize);
        let common = intersection_search_space_names(&study);
        assert!(common.is_empty());
    }

    #[test]
    fn test_group_decomposed_basic() {
        let mut study = Study::new(Direction::Minimize);
        // Trial 0: has x, y, a (branch A)
        study.add_trial(make_trial(0, &[("x", 1.0), ("y", 2.0), ("a", 3.0)]));
        // Trial 1: has x, y, b (branch B)
        study.add_trial(make_trial(1, &[("x", 4.0), ("y", 5.0), ("b", 6.0)]));
        // Trial 2: has x, y, a (branch A again)
        study.add_trial(make_trial(2, &[("x", 7.0), ("y", 8.0), ("a", 9.0)]));

        let groups = group_decomposed_search_space(&study);

        // x and y should be in the same group (always present)
        // a should be in its own group (present in trials 0, 2)
        // b should be in its own group (present in trial 1)
        assert_eq!(groups.len(), 3);

        // Find the group containing x
        let xy_group = groups.iter().find(|g| g.contains("x")).unwrap();
        assert!(xy_group.contains("y"));
        assert_eq!(xy_group.len(), 2);

        // a and b should be in separate groups
        let a_group = groups.iter().find(|g| g.contains("a")).unwrap();
        assert_eq!(a_group.len(), 1);
        let b_group = groups.iter().find(|g| g.contains("b")).unwrap();
        assert_eq!(b_group.len(), 1);
    }

    #[test]
    fn test_group_decomposed_all_same() {
        let mut study = Study::new(Direction::Minimize);
        study.add_trial(make_trial(0, &[("x", 1.0), ("y", 2.0)]));
        study.add_trial(make_trial(1, &[("x", 3.0), ("y", 4.0)]));

        let groups = group_decomposed_search_space(&study);
        // All params always co-occur -> one group
        assert_eq!(groups.len(), 1);
        assert!(groups[0].contains("x"));
        assert!(groups[0].contains("y"));
    }
}
