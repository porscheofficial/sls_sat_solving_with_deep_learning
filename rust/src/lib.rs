use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, WeightedIndex};
use std::collections::HashSet;
use rand::prelude::IteratorRandom;
use rayon::prelude::*;
use std::fs::read_to_string;
use varisat::{dimacs::DimacsParser, CnfFormula, Lit}; // 0.7.2

fn clause_is_violated(clause: &[Lit], assignment: &Vec<bool>) -> bool {
    return clause
        .iter()
        .map(|lit: &Lit| lit.is_positive() != assignment[lit.var().index()])
        .all(|x| x);
}

fn resample_clause(
    clause: &[Lit],
    assignment: &mut Vec<bool>,
    rng: &mut StdRng,
    weights: &Vec<f64>,
) -> Vec<usize> {
    let mut flipped_idxs = vec![];
    for lit in clause {
        let idx = lit.var().index();
        let float_assignment = if assignment[idx] { 1.0 } else { 0.0 };
        let flip_prob: f64 = (1.0 - float_assignment) * weights[idx] + float_assignment * (1.0 -weights[idx]);
        let random_value: f64 = rng.gen();
        if random_value < flip_prob {
            flipped_idxs.push(idx)
        }
    }

    if !flipped_idxs.is_empty(){
        for idx in flipped_idxs.clone(){
            assignment[idx] = !assignment[idx]
        }
    }

    flipped_idxs

    // for lit in clause {
    //     let idx = lit.var().index();
    //     assignment[idx] = rng.gen_bool(weights[idx]);
    // }
}

fn flip_literal(clause: &[Lit], assignment: &mut Vec<bool>, rng: &mut StdRng) -> Vec<usize> {
    let idx = clause.choose(rng).unwrap().var().index();
    assignment[idx] = !assignment[idx];
    vec![idx]
}

fn flip_literal_probsat(
    clause: &[Lit],
    assignment: &mut Vec<bool>,
    rng: &mut StdRng,
    weights: &Vec<f64>,
) -> Vec<usize> {
    let mut weights_sample = vec![];
    for lit in clause {
        let idx = lit.var().index();
        let float_assignment = if assignment[idx] { 1.0 } else { 0.0 };
        weights_sample.push(
            (1.0 - float_assignment) * weights[idx] + float_assignment * (1.0 - weights[idx]),
        );
    }
    let dist = WeightedIndex::new(&weights_sample).unwrap();
    let chosen_idx = clause[dist.sample(rng)].var().index();
    assignment[chosen_idx] = !assignment[chosen_idx];
    vec![chosen_idx]
}

fn flip_literal_moser_single(
    clause: &[Lit],
    assignment: &mut Vec<bool>,
    rng: &mut StdRng,
    weights: &Vec<f64>,
) -> Vec<usize> {
    let mut flipped_idxs = vec![];
    for lit in clause {
        let idx = lit.var().index();
        let float_assignment = if assignment[idx] { 1.0 } else { 0.0 };
        let flip_prob: f64 = (1.0 - float_assignment) * weights[idx] + float_assignment * (1.0 -weights[idx]);
        let random_value: f64 = rng.gen();
        if random_value < flip_prob {
            flipped_idxs.push(idx)
        }
    }

    if !flipped_idxs.is_empty(){
        let chosen_idx = flipped_idxs[rng.gen_range(0..flipped_idxs.len())];
        assignment[chosen_idx] = !assignment[chosen_idx];
        flipped_idxs = vec![chosen_idx];
    }

    flipped_idxs
}

#[pyfunction]
fn run_sls_python(
    algo_type_as_str: String,
    path: String,
    weights_initialize: Vec<f64>,
    weights_resample: Vec<f64>,
    nsteps: usize,
    nruns: usize,
    return_trajectories: bool,
    pre_compute_mapping: bool,
    prob_flip_best: f64,
) -> PyResult<(Vec<bool>, Vec<bool>, usize, Vec<usize>, Vec<Vec<usize>>)> {
    let input: String = read_to_string(path).expect("failed to read");
    let implements_read = &input.as_bytes()[..];

    let formula = DimacsParser::parse(implements_read).expect("parse error");

    let algo_type = match algo_type_as_str.as_str() {
        "moser" => AlgoType::Moser,
        "moser_single" => AlgoType::MoserSingle,
        "schoening" => AlgoType::Schoening,
        "probsat" => AlgoType::Probsat,
        _ => panic!("Unknown algorithm type"),
    };

    // println!("algo_type_as_str: {}", algo_type_as_str);

    let (found_solutions, assignment, best_energy, numsteps, trajectories) = run_sls(
        algo_type,
        formula,
        weights_initialize,
        weights_resample,
        nsteps,
        nruns,
        return_trajectories,
        pre_compute_mapping,
        prob_flip_best,
    );

    return Ok((
        found_solutions,
        assignment,
        best_energy,
        numsteps,
        trajectories,
    ));
}

#[pymodule]
fn moser_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sls_python, m)?)?;
    Ok(())
}

enum AlgoType {
    Moser,
    MoserSingle,
    Schoening,
    Probsat,

}

fn run_sls(
    algo_type: AlgoType,
    formula: CnfFormula,
    weights_initialize: Vec<f64>,
    weights_resample: Vec<f64>,
    nsteps: usize,
    nruns: usize,
    return_trajectories: bool,
    pre_compute_mapping: bool,
    prob_flip_best: f64,
) -> (Vec<bool>, Vec<bool>, usize, Vec<usize>, Vec<Vec<usize>>) {
    let find_violated_clauses = |assignment: &Vec<bool>| {
        formula
            .iter()
            .filter(|clause| clause_is_violated(clause, assignment))
            .collect::<HashSet<_>>()
    };

    let mut trajectories: Vec<Vec<usize>> = vec![];
    let mut num_steps: Vec<usize> = vec![];
    let mut found_solutions: Vec<bool> = vec![];
    let mut var_to_clauses: Vec<Vec<&[Lit]>> = vec![vec![]; formula.var_count()];
    if pre_compute_mapping{
        // let start = time::Instant::now();
        
        for clause in formula.iter() {
            for lit in clause.iter() {
                let var_index = lit.var().index();
                var_to_clauses[var_index].push(clause);
            }
        }
        // let end = time::Instant::now();

        // println!("Time for mapping: {:?}", end - start);
    }
    // let start = time::Instant::now();
    let results: Vec<_> = (0..nruns).into_par_iter().map(|_| {
        let mut trajectory: Vec<usize> = vec![];
        let mut best_energy_run: usize = formula.len();
        let mut best_assignment_run = vec![];
        let mut found_solution = false;
        let seed: u64 = rand::random();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut assignment = (0..formula.var_count())
            .map(|i| rng.gen_bool(weights_initialize[i]))
            .collect();

        let mut numstep_local = 0;
        let mut violated_clauses = find_violated_clauses(&assignment);
        let mut numfalse = violated_clauses.len();

        if numfalse < best_energy_run {
            best_energy_run = numfalse;
            best_assignment_run = assignment.clone();
        }

        if return_trajectories {
            trajectory.push(numfalse);
        }
        
        while numfalse > 0 && numstep_local < nsteps {
            
            numstep_local += 1;
            
            let next_clause = violated_clauses.iter().choose(&mut rng).unwrap();
            
            let flipped_var = if rng.gen::<f64>() < prob_flip_best  {
                // Flip the index in next_clause that leads to the configuration with the lowest number of violated clauses
                let mut best_idx = 0;
                let mut min_violations = usize::MAX;
            
                for lit in *next_clause {
                    let idx = lit.var().index();
                    // Flip the assignment temporarily
                    assignment[idx] = !assignment[idx];
                    let mut violations = 0;
            
                    if pre_compute_mapping{
                        for &clause in &var_to_clauses[idx] {
                            if clause_is_violated(clause, &assignment) {
                                violations += 1;
                            }
                        }
                    } else {
                        violations = find_violated_clauses(&assignment).len();
                    }
                    // for &clause in &var_to_clauses[idx] {
                    //     if clause_is_violated(clause, &assignment) {
                    //         violations += 1;
                    //     }
                    // }
            
                    // Flip it back
                    assignment[idx] = !assignment[idx];
            
                    if violations < min_violations {
                        min_violations = violations;
                        best_idx = idx;
                    }
                }
            
                // Flip the best index permanently
                assignment[best_idx] = !assignment[best_idx];
                vec![best_idx]
            } else {
                match algo_type {
                    AlgoType::Moser => resample_clause(next_clause, &mut assignment, &mut rng, &weights_resample),
                    AlgoType::MoserSingle => flip_literal_moser_single(next_clause, &mut assignment, &mut rng, &weights_resample),
                    AlgoType::Schoening => flip_literal(next_clause, &mut assignment, &mut rng),
                    AlgoType::Probsat => flip_literal_probsat(next_clause, &mut assignment, &mut rng, &weights_resample),
                }
            };
             
            if pre_compute_mapping{
                for flipped_idx in flipped_var {
                    for &clause in &var_to_clauses[flipped_idx] {
                        if clause_is_violated(clause, &assignment) {
                            violated_clauses.insert(clause);
                        } else {
                            violated_clauses.remove(clause);
                        }
                    }
                }
            } else {
                violated_clauses = find_violated_clauses(&assignment);
            }
            
            numfalse = violated_clauses.len();
            if numfalse < best_energy_run {
                best_energy_run = numfalse;
                best_assignment_run = assignment.clone();
            }
            
            if return_trajectories {
                trajectory.push(numfalse);
            }
        }



        if numfalse == 0 {
            found_solution = true;
        }
    

        (
            found_solution,
            best_assignment_run,
            best_energy_run,
            numstep_local,
            trajectory,
        )
    }).collect();
    // let end = time::Instant::now();
    // println!("Time for steps: {:?}", end - start);
    let mut best_energy = formula.len();
    let mut best_assignment = vec![];
    for (found_solution, assignment, numfalse, numstep, trajectory) in results {
        if numfalse < best_energy {
            best_energy = numfalse;
            best_assignment = assignment;
        }
        if return_trajectories {
            trajectories.push(trajectory);
        }
        num_steps.push(numstep);
        found_solutions.push(found_solution);
    }
    // println!("best_energy: {}", best_energy);
    (best_assignment, found_solutions, best_energy, num_steps, trajectories)
}
