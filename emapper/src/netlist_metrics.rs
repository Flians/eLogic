//! Common physical-instance area and unloaded pin-delay measurement for emitted netlists.
use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::techlib::{Cell, TechLib};

#[derive(Debug)]
pub struct NetlistMetrics {
    pub area: f64,
    pub delay: f64,
    pub instances: usize,
    pub multioutput_instances: usize,
}

pub fn library_json(tech: &TechLib) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&tech.cells)
}

/// Read an unflattened Yosys JSON design with a top named `mapped` and library black boxes.
/// Assigns are wire aliases; every physical cell is charged once, including output buffers.
pub fn measure_json(json: &str, tech: &TechLib) -> Result<NetlistMetrics, String> {
    let document: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let top = &document["modules"]["mapped"];
    let cells = top["cells"].as_object().ok_or("missing mapped cells")?;
    let ports = top["ports"].as_object().ok_or("missing mapped ports")?;
    let mut library: HashMap<&str, Vec<&Cell>> = HashMap::new();
    for cell in &tech.cells {
        library.entry(&cell.name).or_default().push(cell);
    }
    let mut drivers: HashMap<u64, (&Value, &Cell)> = HashMap::new();
    let mut inputs = HashSet::new();
    let mut outputs = Vec::new();
    for port in ports.values() {
        let bits = port["bits"].as_array().ok_or("invalid port bits")?;
        match port["direction"].as_str() {
            Some("input") => {
                for bit in bits {
                    inputs.insert(bit.as_u64().ok_or("non-wire input")?);
                }
            }
            Some("output") => outputs.extend(bits),
            _ => return Err("unsupported port direction".into()),
        }
    }
    let mut area = 0.0;
    let mut multioutput_instances = 0;
    for instance in cells.values() {
        let name = instance["type"].as_str().ok_or("invalid cell type")?;
        let group = library
            .get(name)
            .ok_or_else(|| format!("unknown physical cell {name}"))?;
        if group
            .iter()
            .any(|cell| (cell.area - group[0].area).abs() > 1e-6)
        {
            return Err(format!("inconsistent physical area for {name}"));
        }
        area += f64::from(group[0].area);
        multioutput_instances += usize::from(group.len() > 1);
        for &cell in group {
            let pins = instance["connections"][&cell.out_pin].as_array();
            if pins.is_none_or(Vec::is_empty) {
                continue;
            }
            let pins = pins.unwrap();
            if pins.len() != 1 {
                return Err("non-scalar library output".into());
            }
            let bit = pins[0].as_u64().ok_or("constant driven by cell")?;
            if inputs.contains(&bit) || drivers.insert(bit, (instance, cell)).is_some() {
                return Err(format!("multiple drivers for net {bit}"));
            }
        }
    }
    fn arrival(
        bit: &Value,
        inputs: &HashSet<u64>,
        drivers: &HashMap<u64, (&Value, &Cell)>,
        cache: &mut HashMap<u64, f64>,
        active: &mut HashSet<u64>,
    ) -> Result<f64, String> {
        if let Some(constant) = bit.as_str() {
            return match constant {
                "0" | "1" => Ok(0.0),
                _ => Err("unknown or high-impedance signal in measured cone".into()),
            };
        }
        let bit = bit.as_u64().ok_or("invalid signal")?;
        if inputs.contains(&bit) {
            return Ok(0.0);
        }
        if let Some(&time) = cache.get(&bit) {
            return Ok(time);
        }
        if !active.insert(bit) {
            return Err("cyclic physical netlist".into());
        }
        let (instance, cell) = drivers
            .get(&bit)
            .ok_or_else(|| format!("undriven net {bit}"))?;
        let mut time: f64 = 0.0;
        for pin in &cell.vars {
            let connection = instance["connections"][pin]
                .as_array()
                .ok_or("missing input pin")?;
            if connection.len() != 1 {
                return Err("unconnected or non-scalar input pin".into());
            }
            time = time.max(
                arrival(&connection[0], inputs, drivers, cache, active)?
                    + f64::from(cell.pin_delay.get(pin).copied().unwrap_or(0.0)),
            );
        }
        active.remove(&bit);
        cache.insert(bit, time);
        Ok(time)
    }
    let mut cache = HashMap::new();
    let mut active = HashSet::new();
    let mut delay: f64 = 0.0;
    for bit in outputs {
        delay = delay.max(arrival(bit, &inputs, &drivers, &mut cache, &mut active)?);
    }
    // Reject disconnected cyclic/undriven cones too, rather than silently measuring a partial DAG.
    for &bit in drivers.keys() {
        arrival(
            &Value::from(bit),
            &inputs,
            &drivers,
            &mut cache,
            &mut active,
        )?;
    }
    Ok(NetlistMetrics {
        area,
        delay,
        instances: cells.len(),
        multioutput_instances,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tech() -> TechLib {
        TechLib::from_genlib(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/egraph_map.genlib"),
            3,
        )
        .unwrap()
    }

    #[test]
    fn grouped_cell_is_charged_once_and_secondary_pin_is_timed() {
        let json = r#"{"modules":{"mapped":{"ports":{"a":{"direction":"input","bits":[2]},"b":{"direction":"input","bits":[3]},"c":{"direction":"input","bits":[4]},"s":{"direction":"output","bits":[5]},"co":{"direction":"output","bits":[6]}},"cells":{"fa":{"type":"FA","connections":{"A":[2],"B":[3],"C":[4],"S":[5],"CO":[6]}}}}}}"#;
        let value = measure_json(json, &tech()).unwrap();
        assert_eq!(value.area, 2.0);
        assert_eq!(value.delay, 7.0);
        assert_eq!(value.multioutput_instances, 1);
    }

    #[test]
    fn cycle_and_missing_driver_are_rejected() {
        let json = r#"{"modules":{"mapped":{"ports":{"y":{"direction":"output","bits":[3]}},"cells":{"inv":{"type":"INV","connections":{"A":[3],"Y":[3]}}}}}}"#;
        assert!(measure_json(json, &tech()).is_err());
        let json = r#"{"modules":{"mapped":{"ports":{"y":{"direction":"output","bits":[3]}},"cells":{}}}}"#;
        assert!(measure_json(json, &tech()).is_err());
    }
}
