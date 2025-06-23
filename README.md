# uvir

A Rust library for creating MLIR-compatible intermediate representations (IR) with efficient type erasure and static dispatch. Build custom compiler IRs for domain-specific languages with minimal overhead.

## Features

- **MLIR-Compatible**: Parse and print standard MLIR syntax while supporting custom dialects
- **Zero-Cost Abstractions**: Type erasure via static vtables, no runtime boxing overhead
- **Efficient Storage**: String/type interning, slotmap-based regions for fast IR manipulation
- **Extensible Dialects**: Define custom types, attributes, and operations with derive macros
- **Pattern Rewriting**: Built-in pass infrastructure for IR transformations
- **egg Integration**: Seamless interop with the egg e-graph library for advanced optimizations

## Quick Start

```rust
use uvir::prelude::*;

// Define a custom operation
#[derive(Op)]
#[operation(dialect = "myapp", name = "compute")]
struct ComputeOp {
    #[_def]
    result: Val,
    #[_use]
    input: Val,
    #[_attr]
    scale: Attribute,
}

// Parse MLIR text
let mlir = r#"
    func.func @main(%arg0: i32) -> i32 {
        %0 = arith.constant 2 : i32
        %1 = arith.muli %arg0, %0 : i32
        func.return %1 : i32
    }
"#;

let mut ctx = Context::new();
let module = ctx.parse_mlir(mlir)?;
```

## Documentation

- [Getting Started](doc/getting-started.md) - Tutorial and examples
- [Architecture](doc/architecture.md) - Design principles and system overview
- [Type System](doc/type-system.md) - Type erasure and interning details
- [Operations](doc/operations.md) - Operation infrastructure and derive macros
- [Dialects](doc/dialects.md) - Creating custom dialects
- [Pass Infrastructure](doc/passes.md) - Writing IR transformations
- [egg Integration](doc/egg-integration.md) - E-graph optimization support

## Built-in Dialects

uvir includes implementations of core MLIR dialects:

- **builtin**: Core IR constructs (module, unrealized_conversion_cast)
- **arith**: Integer/float arithmetic (30+ operations)
- **scf**: Structured control flow (for, while, if)
- **func**: Function definitions and calls
- **affine**: Affine loop optimizations

## Installation

```toml
[dependencies]
uvir = "0.1"
```

## License

Licensed under either of Apache License 2.0 or MIT license at your option.