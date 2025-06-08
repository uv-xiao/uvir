# uvir

A Rust library for creating new intermediate representations (IR) for DSLs, inspired by MLIR's design principles. 

## TODO

1. Parse/print is not fully MLIR-compatible. 
   - Refer to https://mlir.llvm.org/docs/LangRef/ for the grammar except Block and Branch operations. Summarize the grammar in GRAMMAR.md.
   - For printer_test, rewrite assertions to strictly compare the output with the expected one in MLIR format.
   - For parser_test, add a test with a MLIR snippet to parse.
   - Fix the parser and printer to pass the test.
2. Derive-based operation definition is not tested. 
   - Add a test for it.
   - Fix the implementation to pass the test.
   - Use it for dialects.
3. Operation check is not tested.
4. Finish the pass system and test it. 
5. Add a test for the pass system.

## Progress

- [x] Core Infrastructure: String interning, type system with type erasure, operation infrastructure with static dispatch.
- [x] Memory Management: Efficient slotmap-based storage for regions, values, and operations
- [x] Type System: Type interning with support for builtin types (integer, float, function) and dialect-specific types
- [x] Operation System: Registry-based operation management with static dispatch
- [x] Attribute System: Flexible attribute storage supporting both builtin and dialect-specific attributes
- [x] Dialect Support: Basic arithmetic dialect with constant, add, and multiply operations
- [x] Testing: Basic tests verifying core functionality
- [ ] Parse/print: Basic parse/print functionality with MLIR compatible grammar.

## Core Infrastructure

<details>
<summary>Original proposal</summary>


```rust
pub trait Type {
  fn parser(&self, &mut Parser) -> Result<Self>;
  fn printer(&self, &mut Printer) -> Result<()>;
  // ...
}
pub type DynType = Box<dyn Type>;
pub trait Attribute {}
pub type DynAttribute = Box<dyn Attribute>;
pub struct Value {
  name: Option<String>,
  ty: DynType,
}
pub struct Context {
  global: Region,
}
pub type DynOp = Box<dyn Op>;
pub struct Region {
  values: Slotmap<Val, Value>,
  operations: Slotmap<Opr, DynOp>,
}
pub trait Op {
  fn traits(&self) -> &[&str];
  fn dialect(&self) -> &str;
  fn name(&self) -> &str;
  fn parser(&self, &mut Parser) -> Result<Self>;
  fn printer(&self, &mut Printer) -> Result<()>;
  fn verify(&self) -> Result<()>;
  fn get_defs(&self) -> &[Val];
  fn get_uses(&self) -> &[Val];
  fn get_def(&self, index: usize) -> Val;
  fn get_use(&self, index: usize) -> Val;
  // ...
}
pub struct Parser {}
pub struct Printer {}
```

</details>

### Type System with Type Erasure

Replace `Box<dyn Type>` with an interned type system for better performance and memory efficiency:

```rust
// Type handle - cheap to copy and compare
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(u32);

// Type storage with static dispatch
pub enum TypeKind {
    // Builtin types
    Integer { width: u32, signed: bool },
    Float { precision: FloatPrecision },
    Function { inputs: Vec<TypeId>, outputs: Vec<TypeId> },
    // Dialect-specific types stored as type-erased data
    Dialect { 
        dialect: StringId,
        data: TypeStorage,
    },
}

// Type-erased storage for dialect types
pub struct TypeStorage {
    data: SmallVec<[u8; 16]>, // Small buffer optimization
    vtable: &'static TypeVTable,
}

// Virtual dispatch table for dialect types
pub struct TypeVTable {
    type_id: std::any::TypeId,
    parse: fn(&mut Parser) -> Result<TypeStorage>,
    print: fn(&TypeStorage, &mut Printer) -> Result<()>,
    clone: fn(&TypeStorage) -> TypeStorage,
    eq: fn(&TypeStorage, &TypeStorage) -> bool,
}

// User-facing trait for custom types
pub trait DialectType: Sized + Clone + PartialEq + 'static {
    fn parse(parser: &mut Parser) -> Result<Self>;
    fn print(&self, printer: &mut Printer) -> Result<()>;
}

// Context manages type interning
impl Context {
    pub fn intern_type(&mut self, kind: TypeKind) -> TypeId {
        // Deduplication via HashMap lookup
    }
}
```

### Operation Design with Static Dispatch

Replace `Box<dyn Op>` with a registry-based approach:

```rust
// Operation reference - indexes into arena
#[derive(Clone, Copy)]
pub struct OpRef(u32);

// Static operation metadata
pub struct OpInfo {
    dialect: &'static str,
    name: &'static str,
    traits: &'static [&'static str],
    // Function pointers for polymorphic behavior
    verify: fn(&OpData) -> Result<()>,
    parse: fn(&mut Parser) -> Result<OpData>,
    print: fn(&OpData, &mut Printer) -> Result<()>,
}

// Generic operation storage
pub struct OpData {
    info: &'static OpInfo,
    operands: SmallVec<[Val; 2]>,
    results: SmallVec<[Val; 1]>, 
    attributes: AttributeMap,
    regions: SmallVec<[RegionId; 1]>,
    // Operation-specific data
    custom_data: OpStorage,
}

// Type-erased storage for operation-specific fields
pub struct OpStorage {
    data: SmallVec<[u8; 32]>,
    drop_fn: Option<fn(&mut OpStorage)>,
}

// Value with type information
pub struct Value {
    name: Option<StringId>,
    ty: TypeId,
    defining_op: Option<OpRef>,
}

// Slotmap keys for efficient storage
slotmap::new_key_type! {
    pub struct Val;
    pub struct Opr;
}

pub struct Region {
    values: SlotMap<Val, Value>,
    operations: SlotMap<Opr, OpData>,
    // Maintain operation order
    op_order: Vec<Opr>,
}

// Global operation registry
pub struct OpRegistry {
    ops: HashMap<(StringId, StringId), &'static OpInfo>, // (dialect, name) -> info
}

// Register operations at compile time
inventory::collect!(&'static OpInfo);
```

### Attribute System

Improve attribute storage without boxing:

```rust
// Attribute variants
pub enum Attribute {
    // Builtin attributes  
    Integer(i64),
    Float(f64),
    String(StringId),
    Type(TypeId),
    Array(Vec<Attribute>),
    // Dialect-specific
    Dialect {
        dialect: StringId,
        data: AttributeStorage,
    },
}

// Similar to TypeStorage but for attributes
pub struct AttributeStorage {
    data: SmallVec<[u8; 24]>,
    vtable: &'static AttributeVTable,
}

// Efficient attribute map
pub type AttributeMap = SmallVec<[(StringId, Attribute); 4]>;
```

### String Interning

Add string interning for identifiers:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StringId(u32);

pub struct StringInterner {
    strings: Vec<String>,
    lookup: HashMap<String, StringId>,
}

impl Context {
    pub fn intern_string(&mut self, s: &str) -> StringId {
        self.strings.get_or_intern(s)
    }
}
```

## Features

### Define operations with derive-macros

The derive macro should generate efficient code without dynamic dispatch:

```rust
#[derive(Op)]
#[operation(dialect="arith", name="addu", traits=["SameTy"])]
struct AddU {
  #[_def(ty = "u<?>")]
  c: Val,
  #[_use]
  a: Val,
  #[_use]
  b: Val,
} 
```

The macro generates:

```rust
// Generated static info
static ADDU_INFO: OpInfo = OpInfo {
    dialect: "arith",
    name: "addu",
    traits: &["SameTy"],
    verify: addu_verify,
    parse: addu_parse,
    print: addu_print,
};

// Register with inventory
inventory::submit!(&ADDU_INFO);

// Generated conversion functions
impl AddU {
    fn into_op_data(self, ctx: &mut Context) -> OpData {
        let mut data = OpStorage::new();
        data.write(&self);
        OpData {
            info: &ADDU_INFO,
            operands: smallvec![self.a, self.b],
            results: smallvec![self.c],
            attributes: Default::default(),
            regions: Default::default(),
            custom_data: data,
        }
    }
    
    fn from_op_data(op: &OpData) -> Option<&Self> {
        if op.info as *const _ == &ADDU_INFO as *const _ {
            Some(op.custom_data.as_ref())
        } else {
            None
        }
    }
}

// For egg integration
#[derive(Clone, PartialEq, Eq, Hash)]
struct AddUEgg([egg::Id; 2]); // Compact representation
```

### MLIR compatible.

`uvir` conforms to (subset of) MLIR's language reference. The grammar is the same as MLIR's (https://mlir.llvm.org/docs/LangRef/#).

It supports the following features:
- Dialects.
- Regionalized SSA.

Limitations:
- Only IsolatedFromAbove region is supported.
- No support for `block` and `branch` operations. ONLY SUPPORT structured control flow.
- No support for `RegionKind::Graph`. Region must be `SSA`.

### Pass/Rewrite Infrastructure

Efficient pattern matching and rewriting without excessive allocations:

```rust
// Pattern-based rewriting
pub trait RewritePattern: 'static {
    fn benefit(&self) -> usize { 1 } // Priority
    
    fn match_and_rewrite(
        &self,
        op: OpRef,
        rewriter: &mut PatternRewriter,
        ctx: &Context,
    ) -> Result<bool>;
}

// Rewriter with operation tracking
pub struct PatternRewriter<'a> {
    ctx: &'a mut Context,
    worklist: Vec<OpRef>,
    erased: BitSet,
}

impl PatternRewriter<'_> {
    pub fn replace_op(&mut self, op: OpRef, new_ops: &[OpRef]);
    pub fn erase_op(&mut self, op: OpRef);
    pub fn replace_all_uses(&mut self, from: Val, to: Val);
}

// Greedy pattern driver
pub fn apply_patterns_greedy(
    ctx: &mut Context,
    patterns: &[Box<dyn RewritePattern>],
    region: RegionId,
) -> Result<bool> {
    let mut changed = false;
    let mut rewriter = PatternRewriter::new(ctx);
    
    // Sort patterns by benefit
    let mut patterns = patterns.to_vec();
    patterns.sort_by_key(|p| std::cmp::Reverse(p.benefit()));
    
    // Fixed-point iteration
    loop {
        let mut local_changed = false;
        for op in rewriter.ctx.regions[region].op_order.clone() {
            for pattern in &patterns {
                if pattern.match_and_rewrite(op, &mut rewriter, ctx)? {
                    local_changed = true;
                    break;
                }
            }
        }
        if !local_changed { break; }
        changed = true;
    }
    Ok(changed)
}

// Analysis-based passes
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&mut self, ctx: &mut Context) -> Result<PassResult>;
}

pub struct PassResult {
    pub changed: bool,
    pub statistics: HashMap<String, u64>,
}

// Pass manager with dependency resolution
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    dependencies: HashMap<String, Vec<String>>,
}

impl PassManager {
    pub fn add_pass(&mut self, pass: Box<dyn Pass>);
    pub fn run(&mut self, ctx: &mut Context) -> Result<()>;
}
```

### Interface with `egg`

Efficient bidirectional conversion between uvir and egg:

```rust
// Define egg language from operations
define_egg_language! {
    #[derive(Clone, PartialEq, Eq, Hash)]
    enum ArithLang {
        AddU(AddUEgg),
        MulU(MulUEgg),
        Constant(ConstantEgg),
        // ...
    }
}

// Conversion traits
pub trait IntoEgg {
    type EggRepr;
    fn into_egg(&self, conv: &mut UvirToEgg) -> Self::EggRepr;
}

pub trait FromEgg: Sized {
    type EggRepr;
    fn from_egg(egg: &Self::EggRepr, conv: &mut EggToUvir) -> Result<Self>;
}

// Conversion context maintains mappings
pub struct UvirToEgg<'a> {
    ctx: &'a Context,
    val_to_id: HashMap<Val, egg::Id>,
    egraph: &'a mut EGraph<ArithLang, ()>,
}

pub struct EggToUvir<'a> {
    ctx: &'a mut Context,
    id_to_val: HashMap<egg::Id, Val>,
}

// Example usage
impl Context {
    pub fn to_egraph(&self, region: RegionId) -> EGraph<ArithLang, ()> {
        let mut egraph = EGraph::default();
        let mut conv = UvirToEgg::new(self, &mut egraph);
        
        for op in &self.regions[region].op_order {
            conv.convert_op(op);
        }
        
        egraph
    }
    
    pub fn from_egraph(egraph: &EGraph<ArithLang, ()>) -> Result<Self> {
        let mut ctx = Context::new();
        let mut conv = EggToUvir::new(&mut ctx);
        
        for class in egraph.classes() {
            conv.convert_class(class)?;
        }
        
        Ok(ctx)
    }
}
```

## Implementation Guide

### Project Structure

```
uvir/
├── src/
│   ├── lib.rs           # Public API
│   ├── context.rs       # Context and type/string interning
│   ├── types.rs         # Type system implementation  
│   ├── ops.rs           # Operation infrastructure
│   ├── region.rs        # Region and value management
│   ├── pass.rs          # Pass manager and rewriting
│   ├── parser.rs        # MLIR-compatible parser
│   ├── printer.rs       # MLIR-compatible printer
│   ├── dialects/        # Built-in dialects
│   │   ├── builtin.rs
│   │   └── arith.rs
│   └── egg_interop.rs   # egg integration
├── uvir-macros/         # Derive macros
│   ├── src/
│   │   ├── lib.rs
│   │   ├── op_derive.rs
│   │   └── type_derive.rs
│   └── Cargo.toml
├── tests/
├── examples/
└── Cargo.toml
```

### Example: Creating a Simple Arithmetic Dialect

```rust
// Define types
#[derive(DialectType)]
struct IntegerType {
    width: u32,
    signed: bool,
}

// Define operations
#[derive(Op)]
#[operation(dialect = "arith", name = "constant")]
struct ConstantOp {
    #[_def]
    result: Val,
    #[_attr]
    value: Attribute,
}

#[derive(Op)]
#[operation(dialect = "arith", name = "add")]
struct AddOp {
    #[_def]
    result: Val,
    #[_use]
    lhs: Val,
    #[_use]
    rhs: Val,
}

// Register dialect
pub fn register_arithmetic_dialect(registry: &mut DialectRegistry) {
    registry.register("arith", |r| {
        r.add_type::<IntegerType>();
        r.add_op::<ConstantOp>();
        r.add_op::<AddOp>();
    });
}
```

## Builtins

Dialects to implement:
- [ ] `builtin`: Core types and operations
- [ ] `arith`: Integer and floating-point arithmetic
- [ ] `scf`: Structured control flow (for, while, if)
- [ ] `func`: Function definitions and calls
- [ ] `memref`: Memory references (if needed)
