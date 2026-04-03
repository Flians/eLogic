#[cfg(test)]
mod comparison_tests {
    use super::super::*; // Access lib.rs items
    use std::time::Instant;

    fn run_extraction(
        s: &str,
        extractor_type: &str,
        first_depth: bool,
    ) -> (ffi::CCost, std::time::Duration) {
        let all_rules = rules();
        let expr: egg::RecExpr<MIG> = s.parse().unwrap();
        let vars_default: [u32; 26] = [0; 26];
        let vars_: &[u32] = &vars_default;

        let mut runner = egg::Runner::default()
            .with_expr(&expr)
            .with_iter_limit(15)
            .with_node_limit(10000)
            .with_time_limit(std::time::Duration::from_secs(1));

        let root_id = runner.roots[0];
        runner = runner.run(&all_rules);
        let saturated_egraph = runner.egraph;

        let serialized_egraph = egg_to_serialized_egraph(
            &saturated_egraph,
            &MIGCostFn_dsi::new(&saturated_egraph, vars_, Some(first_depth)),
            root_id,
        );

        let start = Instant::now();
        let (dag_cost_size, dag_cost_depth, aft_expr) = if extractor_type == "unique" {
            let extractor =
                extract::unique_node_extractor::UniqueNodeExtractor::new(Some(first_depth));
            let extraction_result =
                extractor.extract(&serialized_egraph, &serialized_egraph.root_eclasses);
            (
                extraction_result.dag_cost_size(
                    &serialized_egraph,
                    &serialized_egraph.root_eclasses,
                    Some(first_depth),
                ),
                extraction_result.dag_cost_depth(
                    &serialized_egraph,
                    &serialized_egraph.root_eclasses,
                    Some(first_depth),
                ),
                extraction_result.print_aft_expr(&serialized_egraph),
            )
        } else {
            let extractor =
                extract::faster_greedy_dag::FasterGreedyDagExtractor::new(Some(first_depth));
            let extraction_result =
                extractor.extract(&serialized_egraph, &serialized_egraph.root_eclasses);
            (
                extraction_result.dag_cost_size(
                    &serialized_egraph,
                    &serialized_egraph.root_eclasses,
                    Some(first_depth),
                ),
                extraction_result.dag_cost_depth(
                    &serialized_egraph,
                    &serialized_egraph.root_eclasses,
                    Some(first_depth),
                ),
                extraction_result.print_aft_expr(&serialized_egraph),
            )
        };
        let duration = start.elapsed();

        (
            ffi::CCost {
                aft_expr: vec![aft_expr],
                aft_dep: dag_cost_depth,
                aft_size: dag_cost_size,
                aft_invs: 0,
            },
            duration,
        )
    }

    #[test]
    fn compare_extractors() {
        let test_cases = vec![
            "(M (~ 0) f (M 0 e (M b d (M a c g))))",
            "(M (~ 0) (M (~ e) (M 0 e (M 0 a d)) (M (~ 0) e f)) (M (~ e) (M 0 e (M 0 b (~ d))) (M (~ 0) e (M 0 c d))))",
            "(M (~ 0) (M 0 (M (~ 0) a b) (~ (M 0 (M (~ 0) a b) c))) (M (~ 0) (M 0 c (~ (M 0 (M (~ 0) a b) c))) d))",
            "(M 0 (M (~ 0) f (M b d (M a c e))) (~ (M 0 f (M b d (M a c e)))))",
            "(M (~ 0) (M 0 (~ a) (M g (~ (M (~ 0) (M 0 f (~ (M 0 (M (~ 0) e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))) (~ (M 0 e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))))))) (M 0 (M (~ 0) e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))) (~ (M (~ 0) f (M 0 e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d)))))))) (M (~ 0) b g))) (M 0 a (~ (M g (~ (M (~ 0) (M 0 f (~ (M 0 (M (~ 0) e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))) (~ (M 0 e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))))))) (M 0 (M (~ 0) e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))) (~ (M (~ 0) f (M 0 e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d)))))))) (M (~ 0) b g)))))",
        ];

        println!(
            "{:<10} | {:<10} | {:<10} | {:<10} | {:<10}",
            "Extractor", "Depth", "Size", "Time (us)", "Expr Len"
        );
        println!("{:-<60}", "");

        for (i, case) in test_cases.iter().enumerate() {
            println!("Case {}:", i + 1);

            let (unique_cost, unique_time) = run_extraction(case, "unique", true);
            println!(
                "{:<10} | {:<10} | {:<10} | {:<10} | {:<10}",
                "Unique",
                unique_cost.aft_dep,
                unique_cost.aft_size,
                unique_time.as_micros(),
                unique_cost.aft_expr[0].len()
            );

            let (faster_cost, faster_time) = run_extraction(case, "faster", true);
            println!(
                "{:<10} | {:<10} | {:<10} | {:<10} | {:<10}",
                "Faster",
                faster_cost.aft_dep,
                faster_cost.aft_size,
                faster_time.as_micros(),
                faster_cost.aft_expr[0].len()
            );
            println!("{:-<60}", "");
        }
    }
}
