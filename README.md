# uvir

A Rust library for creating new intermediate representations (IR) for DSLs. 

## Core infrastructure

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

## Features

### Define operations with derive-macros.

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

Defines a new operation `AddU` with the `#[derive(Op)]` macro.  Field attributes `_def` and `_use` are used to define the type of the field and whether it is an input or output. `ty="u<?>"` means that the field is a value of type `u<?>`. `#[operation]` specifies the dialect, name, and traits of the operation.

The defination generates:
- The `AddU` struct implementing the `Op` trait.
- Parser and printer for the operation, which can be disabled by `#[operation(no_parser)]` and `#[operation(no_printer)]` for users to implement their own.
- A `<Name>Egg` struct for `egg` interface.

### MLIR compatible.

`uvir` conforms to (subset of) MLIR's language reference. The grammar is the same as MLIR's (https://mlir.llvm.org/docs/LangRef/#).

It supports the following features:
- Dialects.
- Regionalized SSA.

Limitations:
- Only IsolatedFromAbove region is supported.
- No support for `block` and `branch` operations. ONLY SUPPORT structured control flow.
- No support for `RegionKind::Graph`. Region must be `SSA`.

### Pass/Rewrite infrastructure

Provide a unified interface for passes and rewrites.

```rust
pub trait Pass {
  fn match(op: &DynOp) -> bool;
  fn and_then(self, pass: &dyn Pass) -> Self;
  fn apply(self, ctx: &mut Context) -> Result<()>;
}
pub struct PassManager {}
```

In `Pass`, `and_then` and be used for nesting passes, `match` and `apply` is a rewrite-style pass description. `PassManager` is a pass manager that can be used to run passes on a `Context`.

### Interface with `egg`

The `#[derive(Op)]` macro generates a `<Name>Egg` struct for `egg` interface. Since the `Language` in `egg` should be implemented on an `enum`, we need to select a set of operations to be included in the `Language`. A macro `define_egg_interface` is provided to select the operations and generate the `Language` enum. The macro also generates the conversion functions between the `egg::Language` and the `uvir::Context`.


## Builtins

Dialects:
- [ ] `builtin`: https://mlir.llvm.org/docs/Dialects/Builtin
- [ ] ...
