//! Ordered-function banks with immutable, independently published source-structure records.
//! An unseen structure always runs extraction, even when equivalent candidates are known.
use crate::{emap_cover::{self, MappedCoverChoice, MappedCoverNode}, profile_cache as legacy,
            techlib::TechLib, tt9::{Tt9, tt9_bit_get}};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, BTreeSet}, path::{Path, PathBuf},
          sync::{Arc, Mutex, OnceLock, atomic::{AtomicU64, Ordering}}};

const RECORD_LIMIT: u64 = 4 * 1024 * 1024;
const DISK_LIMIT: u64 = 64 * 1024 * 1024;
const SOURCES_PER_FUNCTION: usize = 64;
const EXTRA_COVERS: usize = 8;
static FUNCTIONS: AtomicU64 = AtomicU64::new(0);
static FILES_LOADED: AtomicU64 = AtomicU64::new(0);
static NEW_STRUCTURES: AtomicU64 = AtomicU64::new(0);
static CROSS_QUERIES: AtomicU64 = AtomicU64::new(0);
static EXTRA_RETURNED: AtomicU64 = AtomicU64::new(0);
static PROOF_REJECTS: AtomicU64 = AtomicU64::new(0);
static STORAGE_SKIPS: AtomicU64 = AtomicU64::new(0);
static EXTRACT_ERRORS: AtomicU64 = AtomicU64::new(0);

/// The complete minterm bit string is self-describing: len == 2^arity.
/// No support compaction, variable permutation or redundant arity field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct OrderedTruth { bits: String }
impl OrderedTruth {
    fn new(tt: &Tt9, expression_arity: usize, cut_arity: usize) -> Option<Self> {
        if expression_arity > cut_arity || cut_arity > 9 { return None; }
        let mask = (1usize << expression_arity) - 1;
        Some(Self { bits: (0..1usize << cut_arity)
            .map(|i| if tt9_bit_get(tt, i & mask) {'1'} else {'0'}).collect() })
    }
    fn arity(&self) -> Option<usize> {
        let len=self.bits.len();
        (len>0 && len<=512 && len.is_power_of_two() && self.bits.bytes().all(|b| b==b'0'||b==b'1'))
            .then(|| len.trailing_zeros() as usize)
    }
    fn id(&self) -> String { legacy::digest(self.bits.as_bytes()) }
}
#[derive(Serialize, Deserialize)]
struct SourceRecord {
    schema: u32,
    namespace: String,
    truth: OrderedTruth,
    structure: String,
    payload_sha256: String,
    extract_error: Option<String>,
    covers: Vec<MappedCoverChoice>,
}
fn payload(covers: &[MappedCoverChoice], error: &Option<String>) -> String {
    legacy::digest(&serde_json::to_vec(&(covers,error)).unwrap())
}
/// Structural/schema checks precede a complete functional proof, including ordered pin wiring.
fn proves(c: &MappedCoverChoice, truth: &OrderedTruth, tech: &TechLib) -> bool {
    let Some(n)=truth.arity() else {return false};
    if !legacy::valid(c,n) {return false}
    let mut functions=Vec::with_capacity(c.graph.nodes.len());
    for node in &c.graph.nodes {
        let f=match node {
            MappedCoverNode::Cell {cell,..} => {
                if cell.n_inputs==0 {
                    let Some(original)=tech.cells.iter().find(|g|g.name==cell.cell_name && g.vars.is_empty()) else {return false};
                    let mut tt=original.tt; if cell.out_neg {tt[0]^=1;} Some(tt)
                } else {tech.variant_tt_wiring(cell)}
            },
            _=>Some([0;8]),
        };
        let Some(f)=f else {return false};functions.push(f);
    }
    let mut values=vec![false;c.graph.nodes.len()];
    for (assignment,expected) in truth.bits.bytes().enumerate() {
        for (i,node) in c.graph.nodes.iter().enumerate() {
            values[i]=match node {
                MappedCoverNode::Input(k)=>(assignment>>k)&1==1,
                MappedCoverNode::Const(v)=>*v,
                MappedCoverNode::Cell {fanins,..}=> {
                    let bits=fanins.iter().enumerate().fold(0,|acc,(k,&j)|acc|((values[j] as usize)<<k));
                    tt9_bit_get(&functions[i],bits)
                }
            };
        }
        if values[c.graph.root]!=(expected==b'1') {return false}
    }
    true
}
fn read(path: &Path, namespace: &str, truth: &OrderedTruth, tech: &TechLib) -> Option<SourceRecord> {
    if std::fs::metadata(path).ok()?.len()>RECORD_LIMIT {return None}
    let record:SourceRecord=serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    if record.schema!=2 || record.namespace!=namespace || record.truth!=*truth ||
       path.file_stem()?.to_str()?!=record.structure ||
       record.payload_sha256!=payload(&record.covers,&record.extract_error) ||
       record.covers.iter().any(|c|!proves(c,truth,tech)) {return None}
    Some(record)
}
#[derive(Default)]
struct Bank {
    // Each structure retains its own original anchors in original order.
    sources: BTreeMap<String,Vec<String>>,
    candidates: BTreeMap<String,Arc<MappedCoverChoice>>,
}
impl Bank {
    fn insert(&mut self, structure: String, covers: Vec<MappedCoverChoice>) {
        let mut anchors=Vec::new();
        for c in covers {
            let id=legacy::digest(&serde_json::to_vec(&c.graph).unwrap());
            anchors.push(id.clone());
            self.candidates.entry(id).or_insert_with(||Arc::new(c));
        }
        self.sources.insert(structure,anchors);
    }
    fn select(&self, structure: &str, extras: usize) -> Vec<MappedCoverChoice> {
        let anchors=&self.sources[structure];
        let mut selected=BTreeSet::new();
        let mut out=Vec::new();
        for id in anchors {if selected.insert(id.clone()) {out.push((*self.candidates[id]).clone());}}
        let mut others=self.candidates.iter().filter(|(id,_)|!selected.contains(*id)).collect::<Vec<_>>();
        others.sort_by(|(ka,a),(kb,b)|a.area.total_cmp(&b.area).then(a.delay.total_cmp(&b.delay)).then(ka.cmp(kb)));
        let used=others.len().min(extras);
        if used>0 {CROSS_QUERIES.fetch_add(1,Ordering::Relaxed);EXTRA_RETURNED.fetch_add(used as u64,Ordering::Relaxed);}
        out.extend(others.into_iter().take(extras).map(|(_,c)|(**c).clone()));
        out
    }
}
#[derive(Default)]
struct Store {
    banks: BTreeMap<String,Bank>,
    // Lazily measured once per process/namespace, instead of scanning on every write.
    disk_usage: Option<(usize,u64)>,
}
fn write(path:&Path, record:&SourceRecord, usage:&mut Option<(usize,u64)>) -> std::io::Result<bool> {
    let bytes=serde_json::to_vec(record)?;
    if bytes.len() as u64>RECORD_LIMIT {return Ok(false)}
    let folder=path.parent().unwrap();let namespace=folder.parent().unwrap();
    std::fs::create_dir_all(folder)?;
    if usage.is_none() {
        let mut files=0;let mut total=0;
        for dir in std::fs::read_dir(namespace)? {
            let dir=dir?;if !dir.file_type()?.is_dir(){continue}
            for f in std::fs::read_dir(dir.path())? {
                let f=f?;if f.path().extension().is_some_and(|e|e=="json") {files+=1;total+=f.metadata()?.len();}
            }
        }
        *usage=Some((files,total));
    }
    let old=std::fs::metadata(path).map(|m|m.len()).unwrap_or(0);
    let (files,total)=usage.as_mut().unwrap();
    if (old==0 && *files>=2048) || total.saturating_sub(old)+bytes.len() as u64>DISK_LIMIT {return Ok(false)}
    let tmp=path.with_extension(format!("{}.{}.tmp",std::process::id(),legacy::SERIAL.fetch_add(1,Ordering::Relaxed)));
    std::fs::write(&tmp,&bytes)?;
    let result=std::fs::rename(&tmp,path);
    if result.is_err(){let _=std::fs::remove_file(&tmp);}
    result?;
    *files+=usize::from(old==0);*total=total.saturating_sub(old)+bytes.len() as u64;
    Ok(true)
}
fn query(store:&mut Store, expr:&str, truth:&OrderedTruth, tech:&TechLib,
         namespace:&str, disk:Option<&Path>, extras:usize,
         mut build:impl FnMut()->Result<Vec<MappedCoverChoice>,String>) -> Vec<MappedCoverChoice> {
    let key=truth.id();let structure=legacy::digest(expr.as_bytes());
    let folder=disk.map(|p|p.join(&key));
    let mut loaded=false;
    if !store.banks.contains_key(&key) {
        FUNCTIONS.fetch_add(1,Ordering::Relaxed);
        let mut bank=Bank::default();
        if let Some(dir)=folder.as_ref().and_then(|p|std::fs::read_dir(p).ok()) {
            let mut paths=dir.filter_map(Result::ok).map(|e|e.path())
                .filter(|p|p.extension().is_some_and(|s|s=="json")).collect::<Vec<_>>();paths.sort();
            for p in paths.into_iter().take(SOURCES_PER_FUNCTION) {
                if let Some(record)=read(&p,namespace,truth,tech) {
                    bank.insert(record.structure,record.covers);loaded=true;
                    FILES_LOADED.fetch_add(1,Ordering::Relaxed);
                } else {legacy::CORRUPT.fetch_add(1,Ordering::Relaxed);}
            }
        }
        if store.banks.len()>=1024 {store.banks.pop_first();}
        store.banks.insert(key.clone(),bank);
    }
    let bank=store.banks.get_mut(&key).unwrap();
    if bank.sources.contains_key(&structure) {
        if loaded {legacy::DISK_HITS.fetch_add(1,Ordering::Relaxed);}
        else {legacy::MEMORY_HITS.fetch_add(1,Ordering::Relaxed);}
        return bank.select(&structure,extras);
    }
    if !bank.sources.is_empty(){NEW_STRUCTURES.fetch_add(1,Ordering::Relaxed);}
    legacy::BUILDS.fetch_add(1,Ordering::Relaxed);
    let (mut covers,error)=match build() {Ok(c)=>(c,None),Err(e)=>{EXTRACT_ERRORS.fetch_add(1,Ordering::Relaxed);(Vec::new(),Some(e))}};
    covers.retain(|c|if proves(c,truth,tech){true}else{PROOF_REJECTS.fetch_add(1,Ordering::Relaxed);false});
    let record=SourceRecord {schema:2,namespace:namespace.into(),truth:truth.clone(),structure:structure.clone(),
                            payload_sha256:payload(&covers,&error),covers:covers.clone(),extract_error:error};
    let cacheable=bank.sources.len()<SOURCES_PER_FUNCTION;
    // Always preserve a new structure's anchors for this query, even at the bank limit.
    bank.insert(structure.clone(),covers);
    let out=bank.select(&structure,extras);
    if !cacheable {
        bank.sources.remove(&structure);
        // Forget the bank rather than retain unbounded orphan candidate entries.
        store.banks.remove(&key);STORAGE_SKIPS.fetch_add(1,Ordering::Relaxed);
    } else if let Some(folder)=folder {
        match write(&folder.join(format!("{structure}.json")),&record,&mut store.disk_usage) {
            Ok(true)=>(),Ok(false)=>{STORAGE_SKIPS.fetch_add(1,Ordering::Relaxed);},
            Err(_)=>{legacy::WRITE_FAILURES.fetch_add(1,Ordering::Relaxed);}
        }
    }
    out
}
pub(crate) fn pool(expr:&str, n:usize, expression_arity:u8, tt:&Tt9) -> Vec<MappedCoverChoice> {
    static STORE:OnceLock<Mutex<Store>>=OnceLock::new();
    static NAMESPACE:OnceLock<String>=OnceLock::new();
    let tech=crate::profile_techlib_store();
    let Some(truth)=OrderedTruth::new(tt,expression_arity as usize,n) else {return Vec::new()};
    let namespace=NAMESPACE.get_or_init(|| {
        let search=emap_cover::cut_mapping_params();
        legacy::key_with_namespace("ordered-truth-v2",&legacy::namespace(tech),0,legacy::limits(),
            (search.egg_iter_limit,search.egg_node_limit,search.max_cover_nodes))
    });
    let disk=std::env::var_os("MAP_EGG_PROFILE_CACHE_DIR").map(PathBuf::from)
        .map(|p|p.join("ordered-truth-v2").join(namespace));
    query(&mut STORE.get_or_init(Default::default).lock().unwrap(),expr,&truth,tech,namespace,disk.as_deref(),EXTRA_COVERS,
        ||emap_cover::profile::extract(expr,tech,n,legacy::limits()))
}
pub(crate) fn stats()->String {
    format!("tt_functions={} tt_files_loaded={} tt_new_structures={} tt_cross_queries={} tt_extra_candidates={} tt_proof_rejects={} tt_storage_skips={} tt_extract_errors={}",
        FUNCTIONS.load(Ordering::Relaxed),FILES_LOADED.load(Ordering::Relaxed),NEW_STRUCTURES.load(Ordering::Relaxed),
        CROSS_QUERIES.load(Ordering::Relaxed),EXTRA_RETURNED.load(Ordering::Relaxed),PROOF_REJECTS.load(Ordering::Relaxed),
        STORAGE_SKIPS.load(Ordering::Relaxed),EXTRACT_ERRORS.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn truth(expr:&str,n:usize)->OrderedTruth {
        let (m,tt)=crate::emap_egg::sexp_to_tt9(expr).unwrap();
        OrderedTruth::new(&tt,m as usize,n).unwrap()
    }
    fn fixture()->(TechLib,PathBuf) {
        let root=std::env::temp_dir().join(format!("ordered-bank-{}-{}",std::process::id(),legacy::test_serial()));
        std::fs::create_dir_all(&root).unwrap();let lib=root.join("cells.genlib");
        std::fs::write(&lib,"GATE INV 0.1 Y=!A;\nPIN A INV 1 999 1 0 1 0\nGATE AND 0.2 Y=A*B;\nPIN A NONINV 1 999 2 0 2 0\nPIN B NONINV 1 999 3 0 3 0\nGATE NAND 0.15 Y=!(A*B);\nPIN A INV 1 999 2 0 2 0\nPIN B INV 1 999 3 0 3 0\n").unwrap();
        (TechLib::from_genlib_unindexed(lib,9).unwrap(),root)
    }
    #[test]
    fn ordered_truth_is_self_describing_and_keeps_unused_dimensions() {
        assert_eq!(truth("(& a b)",2),truth("(& b a)",2));
        assert_eq!(truth("(~ (~ (& a b)))",2),truth("(& a b)",2));
        assert_ne!(truth("(& a (~ b))",2),truth("(& b (~ a))",2));
        assert_ne!(truth("a",1),truth("a",2));
        for n in 0..=9 {let t=truth("0",n);assert_eq!(t.arity(),Some(n));assert_eq!(t.bits.len(),1<<n);}
        let t=truth("b",9);assert_eq!(t.arity(),Some(9));
        for (i,b) in t.bits.bytes().enumerate(){assert_eq!(b==b'1',(i>>1)&1==1);}
        assert!(OrderedTruth {bits:"000".into()}.arity().is_none());
        assert!(OrderedTruth {bits:"0x".into()}.arity().is_none());
    }
    #[test]
    fn new_structure_builds_then_reloads_without_rebuilding() {
        let (tech,root)=fixture();let disk=root.join("cache");let tt=truth("(& a b)",2);
        let mut store=Store::default();let mut builds=0;
        let mut run=|store:&mut Store,expr:&str|query(store,expr,&tt,&tech,"ns",Some(&disk),8,||{
            builds+=1;emap_cover::profile::extract(expr,&tech,2,legacy::limits())});
        let first=run(&mut store,"(& a b)");assert!(!first.is_empty());
        let second=run(&mut store,"(& b a)");assert!(!second.is_empty());
        assert!(second.iter().all(|c|proves(c,&tt,&tech)));
        let again=run(&mut store,"(& b a)");assert_eq!(serde_json::to_string(&second).unwrap(),serde_json::to_string(&again).unwrap());
        drop(run);assert_eq!(builds,2);assert_eq!(store.banks.len(),1);
        let loaded=query(&mut Store::default(),"(& a b)",&tt,&tech,"ns",Some(&disk),8,||panic!("known structure rebuilt"));
        assert!(loaded.iter().all(|c|proves(c,&tt,&tech)));
        let structures=&store.banks[&tt.id()].sources;assert_eq!(structures.len(),2);
        assert_eq!(std::fs::read_dir(disk.join(tt.id())).unwrap().count(),2);
        // Disk records are immutable per structure, so two different writers' records survive.
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn structure_anchors_survive_sharing_and_corruption_does_not_mark_seen() {
        let (tech,root)=fixture();let disk=root.join("cache");let tt=truth("(& a b)",2);
        let covers=emap_cover::profile::extract("(& a b)",&tech,2,legacy::limits()).unwrap();
        let mut bank=Bank::default();bank.insert("own".into(),covers.clone());
        let mut shared=covers[0].clone();
        let inv=tech.exact_inv_variant().unwrap().clone();
        for _ in 0..2 {
            shared.graph.nodes.push(MappedCoverNode::Cell {cell:inv.clone(),fanins:vec![shared.graph.root]});
            shared.graph.root=shared.graph.nodes.len()-1;
        }
        shared.area+=2.0*tech.inv_area;
        shared.pin_to_root_delay=emap_cover::compute_pin_to_root_delays(&shared.graph,&tech,2);
        shared.delay=shared.pin_to_root_delay.iter().copied().fold(0.,f32::max);
        assert!(proves(&shared,&tt,&tech));
        bank.insert("other".into(),vec![shared]);
        let actual=bank.select("own",8);
        assert_eq!(actual.len(),covers.len()+1);
        assert_eq!(serde_json::to_string(&actual[..covers.len()]).unwrap(),serde_json::to_string(&covers).unwrap());
        query(&mut Store::default(),"(& a b)",&tt,&tech,"ns",Some(&disk),8,||Ok(covers.clone()));
        let path=disk.join(tt.id()).join(format!("{}.json",legacy::digest(b"(& a b)")));
        assert!(read(&path,"wrong-namespace",&tt,&tech).is_none());
        assert!(read(&path,"ns",&truth("(~ (& a b))",2),&tech).is_none());
        let mut record:SourceRecord=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut wrong=covers[0].clone();wrong.graph.nodes=vec![MappedCoverNode::Input(0)];wrong.graph.root=0;
        record.covers=vec![wrong];record.payload_sha256=payload(&record.covers,&record.extract_error);
        std::fs::write(&path,serde_json::to_vec(&record).unwrap()).unwrap();
        assert!(read(&path,"ns",&tt,&tech).is_none(),"a self-consistent checksum is not a Boolean proof");
        let mut rebuilt=false;
        query(&mut Store::default(),"(& a b)",&tt,&tech,"ns",Some(&disk),8,||{rebuilt=true;Ok(covers.clone())});
        assert!(rebuilt);assert!(read(&path,"ns",&tt,&tech).is_some());
        std::fs::remove_dir_all(root).unwrap();
    }
}
