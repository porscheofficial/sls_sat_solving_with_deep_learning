use pyo3::prelude::*;
use rand::rngs::StdRng;
// use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, WeightedIndex};

use std::env;
use std::fs::read_to_string;
use varisat::{dimacs::DimacsParser, CnfFormula, Lit}; // 0.7.2

fn clause_is_violated(clause: &[Lit], assignment: &Vec<bool>) -> bool {
    return clause
        .iter()
        .map(|lit: &Lit| lit.is_positive() != assignment[lit.var().index()])
        .all(|x| x);
}

// fn choose_next_clause(
//     violated_clauses: &Vec<&[Lit]>,
//     current: Option<&[Lit]>,
//     rng: &mut ThreadRng,
// ) -> Option<&[Lit]> {
//     match current {
//         Some(clause) => violated_clauses
//             .iter()
//             .filter(|&other| other.iter().any(|lit| current.unwrap().contains(&lit)))
//             .collect::<Vec<&[Lit]>>()
//             .choose(rng),
//         _ => violated_clauses.choose(rng),
//     }
// }

fn resample_clause(
    clause: &[Lit],
    assignment: &mut Vec<bool>,
    rng: &mut StdRng,
    weights: &Vec<f64>,
) {
    for lit in clause {
        let idx = lit.var().index();
        assignment[idx] = rng.gen_bool(weights[idx]);
    }
}

fn flip_literal(clause: &[Lit], assignment: &mut Vec<bool>, rng: &mut StdRng) {
    let idx = clause.choose(rng).unwrap().var().index();
    assignment[idx] = !assignment[idx];
}

fn flip_literal_probsat(clause: &[Lit], assignment: &mut Vec<bool>, rng: &mut StdRng, weights: &Vec<f64>) {
    // let mut weights_sample = vec![0.0; clause.len()];
    let mut weights_sample = vec![];
    // println!("length of clause {}", clause.len());
    // println!("weights {}", weights.len());
    // let mut i = 1;
    for lit in clause {
        let idx = lit.var().index();
        //println!("assignment {}",assignment[idx]);
        let float_assignment = if assignment[idx] { 1.0 } else { 0.0 };
        // weights_sample.push(float_assignment * weights[idx] + (1.0 - float_assignment) * (1.0 - weights[idx]));
        weights_sample.push((1.0 - float_assignment) * weights[idx] + float_assignment * (1.0 - weights[idx]));
        //println!("{}", idx);
        // i = i + 1;
    }
    let dist = WeightedIndex::new(&weights_sample).unwrap();
    let chosen_idx = clause[dist.sample(rng)].var().index();
    // println!("chosen_idx {}", chosen_idx);
    // let chosen_idx = clause.choose(rng).unwrap().var().index();
    assignment[chosen_idx] = !assignment[chosen_idx];
}

#[pyfunction]
fn run_sls_python(
    algo_type_as_str: String,
    path: String,
    weights: Vec<f64>,
    nsteps: usize,
    nruns: usize,
    seed: usize,
    return_trajectories: bool,
) -> PyResult<(Vec<bool>, Vec<bool>, usize, Vec<usize>, Vec<Vec<usize>>)> {
    let input: String = read_to_string(path).expect("failed to read");

    let implements_read = &input.as_bytes()[..];

    let formula = DimacsParser::parse(implements_read).expect("parse error");

    // println!(
    //     "Loading problem with {} variables and {} clauses",
    //     formula.var_count(),
    //     formula.len()
    // );

    let algo_type = match algo_type_as_str.as_str() {
        "moser" => AlgoType::Moser,
        "schoening" => AlgoType::Schoening,
        "probsat" => AlgoType::Probsat,
        _ => panic!("Unknown algorithm type"),
    };

    let (found_solutions, assignment, best_energy, numsteps, trajectories) = run_sls(
        algo_type,
        formula,
        weights,
        nsteps,
        nruns,
        seed,
        return_trajectories,
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
    Schoening,
    Probsat
}

fn run_sls(
    algo_type: AlgoType,
    formula: CnfFormula,
    weights: Vec<f64>,
    nsteps: usize,
    nruns: usize,
    seed: usize,
    return_trajectories: bool,
) -> (Vec<bool>, Vec<bool>, usize, Vec<usize>, Vec<Vec<usize>>) {
    // Auxiliary functions
    // assert_eq!(weights.len(), formula.var_count());

    let find_violated_clauses = |assignment: &Vec<bool>| {
        formula
            .iter()
            .filter(|clause| clause_is_violated(clause, assignment))
            .collect()
    };

    let mut rng = StdRng::seed_from_u64(seed as u64);
    // let mut numtry: usize = 0;
    // let mut numstep: usize = 0;
    // let mut found_solution = false;
    let mut best_energy: usize = formula.len();
    let mut best_assignment = vec![];
    let mut trajectories: Vec<Vec<usize>> = vec![];
    let mut num_steps: Vec<usize> = vec![];
    let mut found_solutions: Vec<bool> = vec![];

    for _ in 0..nruns {
        let mut trajectory: Vec<usize> = vec![];

        let mut found_solution = false;
        // numtry += 1;

        let mut assignment = (0..formula.var_count())
            .map(|i| rng.gen_bool(weights[i]))
            .collect();

        let mut numstep = 0;
        let mut violated_clauses: Vec<&[Lit]> = find_violated_clauses(&assignment);
        let mut numfalse = violated_clauses.len();

        if numfalse < best_energy {
            best_energy = numfalse;
            best_assignment = assignment.clone();
        }

        if return_trajectories {
            trajectory.push(numfalse);
        }

        while numfalse > 0 && numstep < nsteps {
            numstep += 1;
            let next_clause = violated_clauses.choose(&mut rng).unwrap();
            match algo_type {
                AlgoType::Moser => {
                    resample_clause(next_clause, &mut assignment, &mut rng, &weights)
                }
                AlgoType::Schoening => flip_literal(next_clause, &mut assignment, &mut rng),
                AlgoType::Probsat => flip_literal_probsat(next_clause, &mut assignment, &mut rng, &weights),
            }
            violated_clauses = find_violated_clauses(&assignment);
            numfalse = violated_clauses.len();

            if numfalse < best_energy {
                best_energy = numfalse;
                best_assignment = assignment.clone();
            }

            if return_trajectories {
                trajectory.push(numfalse);
            }
        }

        if return_trajectories {
            trajectories.push(trajectory);
        }

        num_steps.push(numstep);

        if numfalse == 0 {
            found_solution = true;
        }
        found_solutions.push(found_solution);
    }
    // while !found_solution && numtry < nruns {
    //     let mut trajectory: Vec<usize> = vec![];

    //     numtry += 1;

    //     let mut assignment = (0..formula.var_count())
    //         .map(|i| rng.gen_bool(weights[i]))
    //         .collect();

    //     numstep = 0;
    //     let mut violated_clauses: Vec<&[Lit]> = find_violated_clauses(&assignment);
    //     let mut numfalse = violated_clauses.len();

    //     if numfalse < best_energy {
    //         best_energy = numfalse;
    //         best_assignment = assignment.clone();
    //     }

    //     if return_trajectories {
    //         trajectory.push(numfalse);
    //     }

    //     while numfalse > 0 && numstep < nsteps {
    //         numstep += 1;
    //         let next_clause = violated_clauses.choose(&mut rng).unwrap();
    //         match algo_type {
    //             AlgoType::Moser => {
    //                 resample_clause(next_clause, &mut assignment, &mut rng, &weights)
    //             }
    //             AlgoType::Schoening => flip_literal(next_clause, &mut assignment, &mut rng),
    //         }
    //         // resample_clause(next_clause, &mut assignment, &mut rng, &weights);
    //         violated_clauses = find_violated_clauses(&assignment);
    //         numfalse = violated_clauses.len();

    //         if numfalse < best_energy {
    //             best_energy = numfalse;
    //             best_assignment = assignment.clone();
    //         }

    //         if return_trajectories {
    //             trajectory.push(numfalse);
    //         }

    //         found_solution = numfalse == 0;
    //     }

    //     trajectories.push(trajectory);

    //     // println!("Round {} ending with {} violated clauses", numtry, numfalse);
    // }

    return (
        found_solutions,
        best_assignment,
        best_energy,
        num_steps,
        trajectories,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let input: String = read_to_string(&args[1]).expect("failed to read");

    let implements_read = &input.as_bytes()[..];

    let formula = DimacsParser::parse(implements_read).expect("parse error");

    println!(
        "Loading problem with {} variables and {} clauses",
        formula.var_count(),
        formula.len()
    );
    let mut rng = rand::thread_rng();
    let weights = (0..formula.var_count()).map(|_| rng.gen::<f64>()).collect();
    // let weights = vec![0.5; formula.var_count()];

    let nsteps = 1000;
    let nruns = 10;
    let seed = 0;
    let return_trajectories = false;

    let (found_solutions, assignment, _final_energies, _numsteps, _trajectories) = run_sls(
        AlgoType::Moser,
        formula,
        weights,
        nsteps,
        nruns,
        seed,
        return_trajectories,
    );

    println!(
        "last candidate: {:#?}, Found solution: {}",
        assignment,
        found_solutions.last().unwrap()
    );
}

