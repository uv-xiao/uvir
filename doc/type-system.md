# Type System

uvir's type system provides efficient type representation through interning and type erasure, achieving both flexibility and performance.

## Overview

The type system supports both built-in types (integers, floats, functions) and extensible dialect-specific types without runtime overhead.

```rust
// All types are represented by a lightweight handle
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(u32);

// Type data is stored separately and interned
pub enum TypeKind {
    // Built-in types
    Integer { width: u32, signed: bool },
    Float { precision: FloatPrecision },
    Function { inputs: Vec<TypeId>, outputs: Vec<TypeId> },
    
    // Dialect types use type erasure
    Dialect { 
        dialect: StringId,
        data: TypeStorage,
    },
}
```

## Type Interning

All types are interned in the Context to ensure:
- Each unique type has exactly one TypeId
- Type comparison is just integer comparison
- Memory usage is minimized

```rust
impl Context {
    pub fn get_i32_type(&mut self) -> TypeId {
        self.intern_type(TypeKind::Integer { width: 32, signed: true })
    }
    
    pub fn get_f64_type(&mut self) -> TypeId {
        self.intern_type(TypeKind::Float { precision: FloatPrecision::Double })
    }
    
    pub fn get_function_type(&mut self, inputs: &[TypeId], outputs: &[TypeId]) -> TypeId {
        self.intern_type(TypeKind::Function {
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        })
    }
}
```

## Type Erasure for Dialect Types

Dialect-specific types use type erasure with static vtables:

### TypeStorage

```rust
pub struct TypeStorage {
    data: SmallVec<[u8; 16]>,     // Inline storage for small types
    vtable: &'static TypeVTable,   // Static dispatch table
}

impl TypeStorage {
    pub fn new<T: DialectType>(value: T) -> Self {
        let mut data = SmallVec::new();
        unsafe {
            // Write the value directly into the buffer
            let ptr = data.as_mut_ptr() as *mut T;
            std::ptr::write(ptr, value);
            data.set_len(std::mem::size_of::<T>());
        }
        
        Self {
            data,
            vtable: &T::VTABLE,
        }
    }
    
    pub fn as_ref<T: DialectType>(&self) -> Option<&T> {
        if self.vtable.type_id == std::any::TypeId::of::<T>() {
            Some(unsafe { &*(self.data.as_ptr() as *const T) })
        } else {
            None
        }
    }
}
```

### VTable Definition

```rust
pub struct TypeVTable {
    type_id: std::any::TypeId,
    size: usize,
    align: usize,
    
    // Operations
    parse: fn(&mut Parser) -> Result<TypeStorage>,
    print: fn(&TypeStorage, &mut Printer) -> Result<()>,
    clone: fn(&TypeStorage) -> TypeStorage,
    drop: fn(&mut TypeStorage),
    eq: fn(&TypeStorage, &TypeStorage) -> bool,
    hash: fn(&TypeStorage, &mut dyn Hasher),
}
```

### DialectType Trait

User-defined types implement the DialectType trait:

```rust
pub trait DialectType: Sized + Clone + PartialEq + 'static {
    const DIALECT: &'static str;
    const NAME: &'static str;
    
    fn parse(parser: &mut Parser) -> Result<Self>;
    fn print(&self, printer: &mut Printer) -> Result<()>;
}

// The derive macro generates the vtable
#[derive(DialectType)]
#[dialect_type(dialect = "tensor", name = "tensor")]
struct TensorType {
    shape: Vec<i64>,
    element_type: TypeId,
}
```

## Built-in Types

### Integer Types

```rust
// Parameterized integer types
ctx.get_integer_type(32, true);   // i32
ctx.get_integer_type(64, false);  // u64
ctx.get_integer_type(1, false);   // i1 (boolean)

// Common types have shortcuts
ctx.get_i32_type();
ctx.get_i64_type();
ctx.get_i1_type();
```

### Floating-Point Types

```rust
pub enum FloatPrecision {
    Half,      // f16
    Single,    // f32
    Double,    // f64
    Quad,      // f128
    BFloat16,  // bf16
}

ctx.get_f32_type();
ctx.get_f64_type();
ctx.get_bf16_type();
```

### Function Types

```rust
// func.func @add(i32, i32) -> i32
let i32 = ctx.get_i32_type();
let func_type = ctx.get_function_type(&[i32, i32], &[i32]);

// func.func @multi_return(i64) -> (i32, i32)
let i64 = ctx.get_i64_type();
let multi_ret = ctx.get_function_type(&[i64], &[i32, i32]);
```

## Type Parsing and Printing

Types follow MLIR's syntax:

```rust
// Built-in types
"i32"                    // 32-bit signed integer
"f64"                    // 64-bit float
"(i32, i32) -> i32"      // Function type

// Dialect types
"tensor<4x?xf32>"        // Tensor with dynamic dimension
"memref<*xf32>"          // Unranked memref
"!llvm.ptr<i8>"          // LLVM pointer type
```

## Type Constraints

Operations can specify type constraints:

```rust
#[derive(Op)]
#[operation(dialect = "arith", name = "addi")]
struct AddIOp {
    #[_def(type_constraint = "integer")]
    result: Val,
    #[_use(type_constraint = "integer")] 
    lhs: Val,
    #[_use(type_constraint = "integer")]
    rhs: Val,
}
```

Built-in constraints:
- `integer`: Any integer type
- `float`: Any floating-point type  
- `signless_integer`: Integer without sign semantics
- `any`: No constraint

## Type Inference

Types can be inferred from context:

```rust
// Parser can infer result type from operands
%0 = arith.addi %a, %b : i32  // Result type inferred as i32

// Some ops have explicit type inference rules
impl AddIOp {
    fn infer_return_types(&self, operands: &[TypeId]) -> Vec<TypeId> {
        vec![operands[0]]  // Result type same as first operand
    }
}
```

## Performance Characteristics

### Memory Usage
- TypeId: 4 bytes (vs 24+ bytes for String)
- Integer types: ~12 bytes after interning
- Function types: 16 + 4*(inputs+outputs) bytes
- Dialect types: 16-48 bytes typical (with inline storage)

### Performance
- Type comparison: O(1) integer comparison
- Type creation: O(1) with cache hit, O(n) with cache miss
- Type cloning: O(1) for handle, O(n) for deep clone
- Type hashing: O(1) for built-ins, O(n) for complex types

## Best Practices

1. **Intern early**: Create types once and reuse TypeIds
2. **Use built-in types**: Prefer standard types when possible
3. **Small dialect types**: Keep under 16 bytes for inline storage
4. **Avoid deep nesting**: Flatten complex type hierarchies
5. **Cache type queries**: Store TypeIds instead of recreating

## Example: Custom Vector Type

```rust
#[derive(Clone, PartialEq, DialectType)]
#[dialect_type(dialect = "vector", name = "vector")]
struct VectorType {
    shape: SmallVec<[i64; 4]>,
    element_type: TypeId,
}

impl VectorType {
    pub fn get(ctx: &mut Context, shape: &[i64], element: TypeId) -> TypeId {
        let vector = VectorType {
            shape: shape.into(),
            element_type: element,
        };
        ctx.intern_dialect_type(vector)
    }
    
    pub fn parse(parser: &mut Parser) -> Result<Self> {
        parser.parse_less()?;
        let shape = parser.parse_dimensions()?;
        parser.parse_x()?;
        let element_type = parser.parse_type()?;
        parser.parse_greater()?;
        Ok(VectorType { shape, element_type })
    }
    
    pub fn print(&self, printer: &mut Printer) -> Result<()> {
        printer.print("<")?;
        printer.print_dimensions(&self.shape)?;
        printer.print("x")?;
        printer.print_type(self.element_type)?;
        printer.print(">")?;
        Ok(())
    }
}
```