use pyo3::prelude::*;
use rand::{rngs::ThreadRng, Rng};
use std::env;
use std::fs::read_to_string;
use varisat::{dimacs::DimacsParser, CnfFormula, Lit};

fn clause_is_violated(clause: &[Lit], assignment: &Vec<bool>) -> bool {
    for lit in clause {
        let idx = lit.var().index();
        if (lit.is_positive() && assignment[idx] == false)
            || (lit.is_negative() && assignment[idx] == true)
        {
            return true;
        }
    }
    return false;
}

fn resample_clause(
    clause: &[Lit],
    assignment: &mut Vec<bool>,
    rng: &mut ThreadRng,
    weights: &Vec<f64>,
) {
    for lit in clause {
        let idx = lit.var().index();
        assignment[idx] = rng.gen_bool(weights[idx])
    }
}

#[pyfunction]
fn run_moser_python(
    path: String,
    weights: Vec<f64>,
    nsteps: usize,
    nruns: usize,
) -> PyResult<(bool, Vec<bool>, Vec<usize>)> {
    let input: String = read_to_string(path).expect("failed to read");

    let implements_read = &input.as_bytes()[..];

    let formula = DimacsParser::parse(implements_read).expect("parse error");

    println!(
        "Loading problem with {} variables and {} clauses",
        formula.var_count(),
        formula.len()
    );

    let (found_solution, assignment, final_energies) = run_moser(formula, weights, nsteps, nruns);

    return Ok((found_solution, assignment, final_energies));
}

#[pymodule]
fn moser_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_moser_python, m)?)?;
    Ok(())
}

fn run_moser(
    formula: CnfFormula,
    weights: Vec<f64>,
    nsteps: usize,
    nruns: usize,
) -> (bool, Vec<bool>, Vec<usize>) {
    // Auxiliary functions
    assert_eq!(weights.len(), formula.var_count());

    let find_violated_clauses = |assignment: &Vec<bool>| -> Vec<&[Lit]> {
        formula
            .iter()
            .filter(|clause| clause_is_violated(clause, assignment))
            .collect()
    };

    let mut rng = rand::thread_rng();
    let mut numtry: usize = 0;
    let mut found_solution = false;
    let mut assignment: Vec<bool> = vec![];
    let mut final_energies: Vec<usize> = vec![];
    while !found_solution && numtry < nruns {
        numtry += 1;

        assignment = (0..formula.var_count())
            .map(|_| rng.gen_bool(0.5))
            .collect();

        let mut numstep: usize = 0;

        let mut violated_clauses = find_violated_clauses(&assignment);
        let mut numfalse = violated_clauses.len();

        while numfalse > 0 && numstep < nsteps {
            numstep += 1;
            let next_clause: &[Lit] = violated_clauses[0];
            resample_clause(next_clause, &mut assignment, &mut rng, &weights);
            violated_clauses = find_violated_clauses(&assignment);
            numfalse = violated_clauses.len();
            found_solution = numfalse == 0;
        }
        final_energies.push(numfalse);
        println!("Round {} ending with {} violated clauses", numtry, numfalse);
    }

    return (found_solution, assignment, final_energies);
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

    let (found_solution, assignment, _final_energies) = run_moser(formula, weights, nsteps, nruns);

    println!(
        "last candidate: {:#?}, Found solution: {}",
        assignment, found_solution
    );
}
