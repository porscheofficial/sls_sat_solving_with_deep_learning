enum AlgoType {
    Moser,
    Schoening,
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
