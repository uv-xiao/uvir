# Architecture Overview

uvir is designed with performance and extensibility as primary goals. The architecture follows MLIR's proven design patterns while leveraging Rust's type system for zero-cost abstractions.

## Core Design Principles

### 1. Type Erasure with Static VTables

Unlike traditional object-oriented approaches that use dynamic dispatch (`Box<dyn Trait>`), uvir uses static vtables for polymorphism. This eliminates allocation overhead and enables better compiler optimizations.

```rust
// Instead of Box<dyn Type>, we use:
pub struct TypeStorage {
    data: SmallVec<[u8; 16]>,    // Inline storage for small types
    vtable: &'static TypeVTable,  // Static dispatch table
}

pub struct TypeVTable {
    type_id: std::any::TypeId,
    parse: fn(&mut Parser) -> Result<TypeStorage>,
    print: fn(&TypeStorage, &mut Printer) -> Result<()>,
    clone: fn(&TypeStorage) -> TypeStorage,
    eq: fn(&TypeStorage, &TypeStorage) -> bool,
}
```

Benefits:
- No heap allocations for type/attribute storage
- Direct function calls (no vtable lookup)
- Small buffer optimization for common cases
- Type safety via TypeId checking

### 2. Interning Pattern

All strings and types are interned in the Context, providing:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StringId(u32);  // 4 bytes instead of 24+ for String

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(u32);    // 4 bytes instead of complex type data
```

Benefits:
- O(1) equality comparison (just integer comparison)
- Automatic deduplication
- Reduced memory usage
- Cache-friendly data structures

### 3. Handle-Based References

Operations and values use lightweight handles into slotmaps:

```rust
slotmap::new_key_type! {
    pub struct Val;   // Value handle
    pub struct Opr;   // Operation handle
}

pub struct Region {
    values: SlotMap<Val, Value>,
    operations: SlotMap<Opr, OpData>,
    op_order: Vec<Opr>,  // Maintains insertion order
}
```

Benefits:
- Stable references that survive mutations
- Efficient iteration and random access
- Automatic memory management
- No dangling pointers

## System Architecture

### Context: The Central Hub

The Context manages all global state:

```rust
pub struct Context {
    // String interning
    strings: StringInterner,
    
    // Type interning and deduplication
    types: TypeInterner,
    type_storage: SlotMap<TypeId, TypeKind>,
    
    // Global operation registry
    op_registry: OpRegistry,
    
    // Region storage
    regions: SlotMap<RegionId, Region>,
    
    // Top-level module region
    module: RegionId,
}
```

### Operation System

Operations use a data-oriented design that separates static metadata from dynamic data:

```rust
// Static information shared by all instances
pub struct OpInfo {
    dialect: &'static str,
    name: &'static str,
    traits: &'static [&'static str],
    verify: fn(&OpData) -> Result<()>,
    parse: fn(&mut Parser) -> Result<OpData>,
    print: fn(&OpData, &mut Printer) -> Result<()>,
}

// Dynamic per-instance data
pub struct OpData {
    info: &'static OpInfo,
    operands: SmallVec<[Val; 2]>,
    results: SmallVec<[Val; 1]>,
    attributes: AttributeMap,
    regions: SmallVec<[RegionId; 1]>,
    custom_data: OpStorage,  // Type-erased operation-specific data
}
```

### Memory Layout

uvir optimizes for cache efficiency:

1. **Hot data together**: Frequently accessed fields are grouped
2. **Cold data separate**: Parse/print functions in separate vtables
3. **Small buffer optimization**: Common cases avoid heap allocation
4. **Slotmap locality**: Related data stored contiguously

## Dialect System

Dialects are registered at compile time using the inventory crate:

```rust
// In dialect definition
inventory::submit!(&ADDI_OP_INFO);
inventory::submit!(&INTEGER_TYPE_INFO);

// At runtime
pub fn initialize_dialects(ctx: &mut Context) {
    for op_info in inventory::iter::<&'static OpInfo> {
        ctx.register_op(op_info);
    }
    for type_info in inventory::iter::<&'static TypeInfo> {
        ctx.register_type(type_info);
    }
}
```

## Threading Model

uvir follows a single-threaded mutation model:

- Context is not Send/Sync by default
- All mutations go through &mut Context
- Parallel analysis possible with immutable borrows
- Thread-local contexts for parallel compilation

## MLIR Compatibility

uvir maintains compatibility with MLIR's textual format while focusing on a subset:

### Supported Features
- Dialect namespacing
- SSA values and operations  
- Nested regions
- Attributes and types
- Operation traits and constraints

### Limitations by Design
- Only IsolatedFromAbove regions (no implicit captures)
- No block arguments or terminators (structured control flow only)
- No graph regions (SSA only)
- No custom assembly formats (standard format only)

These limitations simplify the implementation while covering most use cases.

## Performance Considerations

### Allocation Strategy
- Pre-allocate common sizes with SmallVec
- Reuse allocations via slotmap compaction
- Intern all strings and types
- Zero-copy parsing where possible

### Cache Efficiency
- Type/string IDs fit in cache lines
- Operations store data inline when possible
- Slotmaps provide spatial locality
- Hot/cold data separation

### Optimization Opportunities
- Static dispatch enables inlining
- Type erasure without boxing
- Compile-time registration
- Minimal pointer chasing