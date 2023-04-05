use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::{rngs::ThreadRng, Rng};

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

#[pyfunction]
fn run_moser_python(
    path: String,
    weights: Vec<f64>,
    nsteps: usize,
    nruns: usize,
    seed: usize,
) -> PyResult<(bool, Vec<bool>, usize, usize, usize)> {
    let input: String = read_to_string(path).expect("failed to read");

    let implements_read = &input.as_bytes()[..];

    let formula = DimacsParser::parse(implements_read).expect("parse error");

    // println!(
    //     "Loading problem with {} variables and {} clauses",
    //     formula.var_count(),
    //     formula.len()
    // );

    let (found_solution, assignment, final_energies, numtry, numstep) =
        run_moser(formula, weights, nsteps, nruns, seed);

    return Ok((found_solution, assignment, final_energies, numtry, numstep));
}

#[pyfunction]
fn run_schoening_python(
    path: String,
    nsteps: usize,
    nruns: usize,
    seed: usize,
) -> PyResult<(bool, Vec<bool>, usize, usize, usize)> {
    let input: String = read_to_string(path).expect("failed to read");

    let implements_read = &input.as_bytes()[..];

    let formula = DimacsParser::parse(implements_read).expect("parse error");

    // println!(
    //     "Loading problem with {} variables and {} clauses",
    //     formula.var_count(),
    //     formula.len()
    // );

    let (found_solution, assignment, final_energies, numtry, numstep) =
        run_schoening(formula, nsteps, nruns, seed);

    return Ok((found_solution, assignment, final_energies, numtry, numstep));
}

#[pymodule]
fn moser_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_moser_python, m)?)?;
    m.add_function(wrap_pyfunction!(run_schoening_python, m)?)?;
    Ok(())
}

fn run_moser(
    formula: CnfFormula,
    weights: Vec<f64>,
    nsteps: usize,
    nruns: usize,
    seed: usize,
) -> (bool, Vec<bool>, usize, usize, usize) {
    // Auxiliary functions
    assert_eq!(weights.len(), formula.var_count());

    let find_violated_clauses = |assignment: &Vec<bool>| {
        formula
            .iter()
            .filter(|clause| clause_is_violated(clause, assignment))
            .collect()
    };

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut numtry: usize = 0;
    let mut found_solution = false;
    let mut best_energy: usize = formula.len();
    let mut best_assignment = vec![];

    while !found_solution && numtry < nruns {
        numtry += 1;

        let mut assignment = (0..formula.var_count())
            .map(|_| rng.gen_bool(0.5))
            .collect();

        let mut numstep: usize = 0;
        let mut violated_clauses: Vec<&[Lit]> = find_violated_clauses(&assignment);
        let mut numfalse = violated_clauses.len();

        if numfalse < best_energy {
            best_energy = numfalse;
            best_assignment = assignment.clone();
        }

        while numfalse > 0 && numstep < nsteps {
            numstep += 1;
            let next_clause = violated_clauses.choose(&mut rng).unwrap();
            resample_clause(next_clause, &mut assignment, &mut rng, &weights);
            violated_clauses = find_violated_clauses(&assignment);
            numfalse = violated_clauses.len();

            if numfalse < best_energy {
                best_energy = numfalse;
                best_assignment = assignment.clone();
            }

            found_solution = numfalse == 0;
        }

        println!("Round {} ending with {} violated clauses", numtry, numfalse);
    }

    return (
        found_solution,
        best_assignment,
        best_energy,
        numtry,
        numstep,
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

    let (found_solution, assignment, _final_energies, _numtry, _numstep) =
        run_moser(formula, weights, nsteps, nruns, seed);

    println!(
        "last candidate: {:#?}, Found solution: {}",
        assignment, found_solution
    );
}
