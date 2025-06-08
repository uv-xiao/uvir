# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Build
```bash
cargo build                    # Build debug version
cargo build --release          # Build release version
cargo build --workspace        # Build all workspace members
```

### Test
```bash
cargo test                     # Run all tests
cargo test --workspace         # Run tests for all workspace members
cargo test [test_name]         # Run specific test by name
cargo test --test [file]       # Run specific test file
cargo test -- --nocapture      # Show println! output during tests
```

### Lint and Format
```bash
cargo fmt                      # Format code using rustfmt
cargo fmt --check              # Check formatting without modifying
cargo clippy                   # Run clippy linter
cargo clippy -- -D warnings    # Treat warnings as errors
```

## Architecture Overview

### Core Design Principles

1. **Type Erasure with VTables**: Both types and attributes use static vtables for polymorphism without runtime overhead. This allows dialect-specific types/attributes while maintaining performance.

2. **Interning Pattern**: Strings and types are interned in the Context for efficiency:
   - StringId/TypeId are 32-bit handles instead of full data
   - Enables O(1) comparison and deduplication
   - All interned data lives in Context

3. **Handle-Based References**: Val/Opr/RegionId are lightweight handles into slotmaps providing:
   - Stable references that survive mutations
   - Efficient iteration and lookup
   - Memory safety without complex lifetimes

### Key Components

**Context** (`context.rs`): Central hub managing all global state including string/type interning, operation registry, and region management.

**Type System** (`types.rs`): Hybrid system with builtin types (integer, float, function) and extensible dialect types via TypeStorage/vtables.

**Operations** (`ops.rs`): Data-oriented design separating static metadata (OpInfo) from dynamic data (OpData). Uses OpStorage for type-erased dialect-specific data.

**Regions** (`region.rs`): Hierarchical structure using SlotMaps for efficient storage of values and operations while maintaining order.

**Pass System** (`pass.rs`): Supports both pattern-based rewriting (RewritePattern) and full IR passes with a PassManager for orchestration.

### MLIR Compatibility

The project aims for MLIR compatibility with these limitations:
- Only IsolatedFromAbove regions supported
- No block/branch operations (structured control flow only)
- No RegionKind::Graph support (SSA only)

### Current TODOs

1. **Parser/Printer**: Make fully MLIR-compatible by fixing tests to use strict MLIR format
2. **Derive Macros**: Test and use the Op derive macro for dialect definitions
3. **Operation Verification**: Add tests for operation checking
4. **Pass System**: Complete implementation and add comprehensive tests

### Testing Strategy

Tests are organized by component in the `tests/` directory. When adding features:
- Unit test individual components
- Integration test interactions between components
- Use MLIR-format snippets for parser/printer tests

### Dialect Development

When creating new dialects:
1. Define types by implementing `DialectType` trait
2. Define operations using `#[derive(Op)]` macro
3. Register with the dialect registry
4. Follow existing patterns in `dialects/arith.rs`