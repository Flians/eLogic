#![allow(warnings)]

use mignite::mig4::Mig;
use mignite::mig4_egg::simplify;
use mignite::mig4_egg::Prop;
use mignite::mig4_map::Mapper;

fn compute_cuts(
    max_cuts: usize,
    max_inputs: usize,
    lut_area: &[u32],
    lut_delay: &[&[i32]],
    wire_delay: i32,
    mig: &Mig,
) {
    let mut depth1_mapper = Mapper::new(max_cuts, max_inputs, lut_area, lut_delay, wire_delay, mig);
    depth1_mapper.compute_cuts(
        Mapper::cut_rank_depth,
        Mapper::cut_rank_size,
        Mapper::cut_rank_area_flow,
    );
    let luts = depth1_mapper.map_luts(true);
    let area = luts
        .iter()
        .map(|cut| lut_area[cut.input_count()])
        .sum::<u32>();
    let mut best = area;
    let mut best_result = luts;

    for _ in 1..=5 {
        depth1_mapper.compute_cuts(
            Mapper::cut_rank_area_flow,
            Mapper::cut_rank_edge_flow,
            Mapper::cut_rank_fanin_refs,
        );
        let luts = depth1_mapper.map_luts(false);
        let area = luts
            .iter()
            .map(|cut| lut_area[cut.input_count()])
            .sum::<u32>();
        if area < best {
            best = area;
            best_result = luts;
            println!("[new best]");
        }
    }

    for _ in 1..=5 {
        depth1_mapper.compute_cuts(
            Mapper::cut_rank_exact_area,
            Mapper::cut_rank_exact_edge,
            Mapper::cut_rank_fanin_refs,
        );
        let luts = depth1_mapper.map_luts(false);
        let area = luts
            .iter()
            .map(|cut| lut_area[cut.input_count()])
            .sum::<u32>();
        if area < best {
            best = area;
            best_result = luts;
            println!("[new best]");
        }
    }

    println!("Mapped to {} LUTs", best_result.len());
    println!(
        "Estimated area: {} units",
        best_result
            .iter()
            .map(|cut| lut_area[cut.input_count()])
            .sum::<u32>()
    );

    for i in 1..=max_inputs {
        println!(
            "LUT{}: {}",
            i,
            best_result
                .iter()
                .filter(|cut| cut.input_count() == i)
                .count()
        );
    }
}

pub(crate) fn to_sexp(nodes: Vec<Prop>) -> Mig {
    let last = nodes.len() - 1;
    let mut mig = Mig::default();
    to_sexp_rec(nodes, last, &mut |_| None);
    mig
}

fn to_sexp_rec(nodes: Vec<Prop>, i: usize, f: &mut impl FnMut(usize) -> Option<String>) {
    let node = &nodes[i];
    let op = node.to_string();
    match node {
        Prop::Maj(..) => 1 as usize,
        Prop::Not(..) => 0 as usize,
        _ => 0 as usize,
    };
    for j in 0..=i {
        let node = &nodes[j];
        let op = node.to_string();
        println!("{:?}", node.to_string());
    }
    println!("{:?}", node);
}

fn main() {
    let best = simplify(&["(M x3 (M x3 x4 (M x5 x6 x7)) x1)"], None);
    to_sexp_rec(best.as_ref().to_vec(), best.as_ref().len() - 1, &mut |_| {
        None
    });
    const UNIT_K: usize = 6;
    const UNIT_C: usize = 8;
    const UNIT_W: i32 = 1;
    const UNIT_LUT_AREA: [u32; 7] = [0, 1, 1, 1, 1, 1, 1];
    const UNIT_LUT_DELAY: [&[i32]; 7] = [
        &[],
        &[0],
        &[0, 0],
        &[0, 0, 0],
        &[0, 0, 0, 0],
        &[0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0, 0],
    ];

    let mut mig = Mig::from_aiger("tests/adder1.aag");

    mig.cleanup_graph();

    println!();
    println!("Unit delay:");
    println!();
    compute_cuts(
        UNIT_C,
        UNIT_K,
        &UNIT_LUT_AREA,
        &UNIT_LUT_DELAY,
        UNIT_W,
        &mig,
    );

    mig.to_graphviz("before.dot").unwrap();

    // mig.optimise_global();
    mig.optimise_area(&mig.input_nodes());

    let f = std::fs::File::create("test.il").unwrap();
    mig.to_rtlil(f).unwrap();

    mig.to_graphviz("after.dot").unwrap();
}
