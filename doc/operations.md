# Operations

Operations are the core building blocks of uvir IR. They represent computations, control flow, and other actions in the program.

## Operation Structure

Each operation consists of:
- **Static metadata** (OpInfo): Shared by all instances of an operation type
- **Dynamic data** (OpData): Per-instance state including operands, results, attributes, and regions

```rust
// Static information - one per operation type
pub struct OpInfo {
    dialect: &'static str,
    name: &'static str,
    traits: &'static [&'static str],
    verify: fn(&OpData) -> Result<()>,
    parse: fn(&mut Parser) -> Result<OpData>,
    print: fn(&OpData, &mut Printer) -> Result<()>,
}

// Dynamic information - one per operation instance
pub struct OpData {
    info: &'static OpInfo,
    operands: SmallVec<[Val; 2]>,
    results: SmallVec<[Val; 1]>,
    attributes: AttributeMap,
    regions: SmallVec<[RegionId; 1]>,
    custom_data: OpStorage,
}
```

## Defining Operations with Derive Macros

The `#[derive(Op)]` macro provides a convenient way to define operations:

```rust
#[derive(Op)]
#[operation(dialect = "arith", name = "addi", traits = ["Commutative", "SameOperandsAndResultType"])]
struct AddIOp {
    #[_def]
    result: Val,
    #[_use]
    lhs: Val,
    #[_use] 
    rhs: Val,
}
```

### Field Attributes

- `#[_def]`: Marks a result value produced by the operation
- `#[_use]`: Marks an operand value consumed by the operation
- `#[_attr]`: Marks an attribute attached to the operation
- `#[_region]`: Marks a nested region owned by the operation

### Operation Attributes

- `dialect`: The dialect namespace (e.g., "arith", "scf")
- `name`: The operation name within the dialect
- `traits`: Operation traits that define behavior and constraints

## Operation Traits

Traits define semantic properties and constraints:

### Common Traits

- **Commutative**: Operands can be reordered (a + b = b + a)
- **Associative**: Operations can be regrouped ((a + b) + c = a + (b + c))
- **Idempotent**: Applying twice has no additional effect
- **Pure**: No side effects, result depends only on operands
- **Terminator**: Must be the last operation in a block
- **IsolatedFromAbove**: Region cannot reference values from enclosing regions

### Type Constraint Traits

- **SameOperandsAndResultType**: All operands and results have the same type
- **SameTypeOperands**: All operands have the same type
- **SameOperandsShape**: Operands have the same shape (for tensor/vector types)

### Example with Constraints

```rust
#[derive(Op)]
#[operation(
    dialect = "arith",
    name = "cmpi",
    traits = ["SameTypeOperands", "Terminator"]
)]
struct CmpIOp {
    #[_def(type_constraint = "i1")]  // Result must be i1 (boolean)
    result: Val,
    
    #[_attr(type_constraint = "cmp_predicate")]
    predicate: Attribute,
    
    #[_use(type_constraint = "integer")]
    lhs: Val,
    
    #[_use(type_constraint = "integer")]
    rhs: Val,
}
```

## Type-Erased Storage

Operation-specific data is stored using type erasure:

```rust
pub struct OpStorage {
    data: SmallVec<[u8; 32]>,
    drop_fn: Option<fn(&mut OpStorage)>,
}

impl OpStorage {
    pub fn new<T: 'static>() -> Self {
        Self {
            data: SmallVec::new(),
            drop_fn: None,
        }
    }
    
    pub fn write<T: 'static>(&mut self, value: &T) {
        unsafe {
            let size = std::mem::size_of::<T>();
            self.data.resize(size, 0);
            std::ptr::copy_nonoverlapping(
                value as *const T as *const u8,
                self.data.as_mut_ptr(),
                size,
            );
        }
    }
    
    pub fn read<T: 'static>(&self) -> &T {
        unsafe { &*(self.data.as_ptr() as *const T) }
    }
}
```

## Operation Registry

Operations are registered at compile time using the inventory crate:

```rust
// In the derive macro expansion
static ADDI_OP_INFO: OpInfo = OpInfo {
    dialect: "arith",
    name: "addi",
    traits: &["Commutative", "SameOperandsAndResultType"],
    verify: addi_verify,
    parse: addi_parse,
    print: addi_print,
};

inventory::submit!(&ADDI_OP_INFO);

// At runtime initialization
pub fn initialize_ops(registry: &mut OpRegistry) {
    for op_info in inventory::iter::<&'static OpInfo> {
        registry.register(op_info);
    }
}
```

## Operation Verification

Operations can define custom verification logic:

```rust
fn addi_verify(op: &OpData) -> Result<()> {
    // Check that we have exactly 2 operands and 1 result
    if op.operands.len() != 2 {
        return Err(Error::InvalidOperandCount);
    }
    if op.results.len() != 1 {
        return Err(Error::InvalidResultCount);
    }
    
    // Type checking is handled by traits
    Ok(())
}

// More complex verification
fn scf_for_verify(op: &OpData) -> Result<()> {
    let data = op.custom_data.read::<ScfForOp>();
    
    // Check loop bounds
    if let (Some(lower), Some(upper)) = (data.lower_bound, data.upper_bound) {
        if lower > upper {
            return Err(Error::InvalidLoopBounds);
        }
    }
    
    // Check that the body region has correct signature
    let body = &op.regions[0];
    if body.num_arguments() != 1 {
        return Err(Error::InvalidRegionSignature);
    }
    
    Ok(())
}
```

## Parsing and Printing

Operations follow MLIR's assembly format:

```mlir
// Basic operation
%0 = arith.addi %a, %b : i32

// With attributes
%1 = arith.constant 42 : i32

// With regions
scf.for %i = %c0 to %c10 step %c1 {
    %2 = arith.muli %i, %i : i32
    scf.yield
}

// Multiple results
%3:2 = arith.add_with_overflow %x, %y : i32
```

## Examples

### Simple Arithmetic Operation

```rust
#[derive(Op)]
#[operation(dialect = "arith", name = "muli", traits = ["Commutative", "SameOperandsAndResultType"])]
struct MulIOp {
    #[_def]
    result: Val,
    #[_use]
    lhs: Val,
    #[_use]
    rhs: Val,
}
```

### Control Flow Operation

```rust
#[derive(Op)]
#[operation(dialect = "scf", name = "if", traits = ["SingleBlock"])]
struct IfOp {
    #[_def]
    results: Vec<Val>,
    
    #[_use(type_constraint = "i1")]
    condition: Val,
    
    #[_region(min_size = 1, max_size = 2)]
    regions: Vec<RegionId>,  // then and optional else regions
}
```

### Operation with Attributes

```rust
#[derive(Op)]
#[operation(dialect = "arith", name = "constant")]
struct ConstantOp {
    #[_def]
    result: Val,
    
    #[_attr(required = true)]
    value: Attribute,
}
```

### Variadic Operation

```rust
#[derive(Op)]
#[operation(dialect = "func", name = "call")]
struct CallOp {
    #[_def]
    results: Vec<Val>,
    
    #[_attr(type_constraint = "symbol_ref")]
    callee: Attribute,
    
    #[_use]
    operands: Vec<Val>,
}
```

## Performance Considerations

### Memory Layout
- OpInfo is shared among all instances (one per operation type)
- OpData uses SmallVec for common cases (1-2 operands/results)
- Custom data is inline when possible (32 bytes)
- Attributes stored in sorted SmallVec for cache efficiency

### Dispatch Performance
- Static dispatch through function pointers
- No virtual method calls
- Direct memory access for custom data
- Trait checking is compile-time when possible

## Best Practices

1. **Use derive macros**: Prefer `#[derive(Op)]` over manual implementation
2. **Minimize custom data**: Keep operation-specific data small
3. **Validate in verify()**: Put complex validation in the verify function
4. **Use standard traits**: Leverage built-in traits for common patterns
5. **Profile your operations**: Measure performance of hot operations

## Advanced Topics

### Custom Traits

```rust
pub trait OpTrait {
    fn verify(op: &OpData) -> Result<()>;
}

// Register custom trait
registry.register_trait("MyTrait", MyTraitImpl::verify);
```

### Operation Folding

```rust
impl AddIOp {
    fn fold(&self, operands: &[Attribute]) -> Option<Attribute> {
        match (operands[0], operands[1]) {
            (Attribute::Integer(a), Attribute::Integer(b)) => {
                Some(Attribute::Integer(a + b))
            }
            _ => None,
        }
    }
}
```

### Custom Builders

```rust
impl AddIOp {
    pub fn build(ctx: &mut Context, lhs: Val, rhs: Val) -> OpData {
        let result_type = ctx.get_value_type(lhs);
        let result = ctx.create_value(result_type);
        
        AddIOp { result, lhs, rhs }.into_op_data(ctx)
    }
}
```