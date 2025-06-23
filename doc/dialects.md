# Dialect Development Guide

Dialects are the primary extension mechanism in uvir, allowing you to define domain-specific types, attributes, and operations.

## What is a Dialect?

A dialect is a namespace that groups related IR constructs:
- **Types**: Data types specific to your domain
- **Attributes**: Compile-time constants and metadata
- **Operations**: Instructions that operate on values

## Creating a New Dialect

### Step 1: Define the Dialect Module

Create a new module for your dialect:

```rust
// src/dialects/mydialect.rs
use uvir::prelude::*;

pub fn register(registry: &mut DialectRegistry) {
    registry.register("mydialect", |dialect| {
        // Register types
        dialect.add_type::<MyCustomType>();
        
        // Register attributes  
        dialect.add_attribute::<MyAttribute>();
        
        // Operations are registered automatically via inventory
    });
}
```

### Step 2: Define Custom Types

```rust
#[derive(Clone, PartialEq, DialectType)]
#[dialect_type(dialect = "mydialect", name = "tensor")]
pub struct TensorType {
    shape: Vec<i64>,
    dtype: TypeId,
}

impl TensorType {
    pub fn get(ctx: &mut Context, shape: Vec<i64>, dtype: TypeId) -> TypeId {
        ctx.intern_dialect_type(TensorType { shape, dtype })
    }
    
    pub fn parse(parser: &mut Parser) -> Result<Self> {
        parser.parse_less()?;
        let shape = parser.parse_dimension_list()?;
        parser.parse_x()?;
        let dtype = parser.parse_type()?;
        parser.parse_greater()?;
        Ok(TensorType { shape, dtype })
    }
    
    pub fn print(&self, printer: &mut Printer) -> Result<()> {
        printer.print("<")?;
        printer.print_dimensions(&self.shape)?;
        printer.print("x")?;
        printer.print_type(self.dtype)?;
        printer.print(">")?;
        Ok(())
    }
}
```

### Step 3: Define Operations

```rust
#[derive(Op)]
#[operation(
    dialect = "mydialect",
    name = "matmul",
    traits = ["Pure"]
)]
pub struct MatMulOp {
    #[_def(type_constraint = "tensor")]
    result: Val,
    
    #[_use(type_constraint = "tensor")]
    lhs: Val,
    
    #[_use(type_constraint = "tensor")]
    rhs: Val,
    
    #[_attr]
    transpose_a: Option<Attribute>,
    
    #[_attr]
    transpose_b: Option<Attribute>,
}

impl MatMulOp {
    pub fn verify(&self, ctx: &Context) -> Result<()> {
        // Get operand types
        let lhs_type = ctx.get_value_type(self.lhs);
        let rhs_type = ctx.get_value_type(self.rhs);
        let result_type = ctx.get_value_type(self.result);
        
        // Extract tensor types
        let lhs_tensor = ctx.get_dialect_type::<TensorType>(lhs_type)?;
        let rhs_tensor = ctx.get_dialect_type::<TensorType>(rhs_type)?;
        let result_tensor = ctx.get_dialect_type::<TensorType>(result_type)?;
        
        // Verify shapes are compatible for matrix multiplication
        // ... shape checking logic ...
        
        Ok(())
    }
}
```

### Step 4: Define Attributes

```rust
#[derive(Clone, PartialEq, DialectAttribute)]
#[dialect_attribute(dialect = "mydialect", name = "layout")]
pub enum LayoutAttr {
    RowMajor,
    ColumnMajor,
    Sparse { format: String },
}

impl LayoutAttr {
    pub fn parse(parser: &mut Parser) -> Result<Self> {
        let layout = parser.parse_keyword()?;
        match layout.as_str() {
            "row_major" => Ok(LayoutAttr::RowMajor),
            "col_major" => Ok(LayoutAttr::ColumnMajor),
            "sparse" => {
                parser.parse_less()?;
                let format = parser.parse_string()?;
                parser.parse_greater()?;
                Ok(LayoutAttr::Sparse { format })
            }
            _ => Err(Error::UnknownLayout(layout)),
        }
    }
    
    pub fn print(&self, printer: &mut Printer) -> Result<()> {
        match self {
            LayoutAttr::RowMajor => printer.print("row_major"),
            LayoutAttr::ColumnMajor => printer.print("col_major"),
            LayoutAttr::Sparse { format } => {
                printer.print("sparse<")?;
                printer.print_string(format)?;
                printer.print(">")
            }
        }
    }
}
```

## Complete Example: Polynomial Dialect

Let's create a complete dialect for polynomial arithmetic:

```rust
// src/dialects/polynomial.rs

#[derive(Clone, PartialEq, DialectType)]
#[dialect_type(dialect = "poly", name = "polynomial")]
pub struct PolynomialType {
    coefficient_type: TypeId,
    degree_bound: Option<u32>,
}

#[derive(Op)]
#[operation(dialect = "poly", name = "constant")]
pub struct PolyConstantOp {
    #[_def(type_constraint = "polynomial")]
    result: Val,
    
    #[_attr(required = true)]
    coefficients: Attribute,  // Array of coefficients
}

#[derive(Op)]
#[operation(
    dialect = "poly",
    name = "add",
    traits = ["Commutative", "Associative", "SameOperandsAndResultType"]
)]
pub struct PolyAddOp {
    #[_def]
    result: Val,
    #[_use]
    lhs: Val,
    #[_use]
    rhs: Val,
}

#[derive(Op)]
#[operation(
    dialect = "poly",
    name = "mul",
    traits = ["Commutative", "Associative"]
)]
pub struct PolyMulOp {
    #[_def]
    result: Val,
    #[_use]
    lhs: Val,
    #[_use]
    rhs: Val,
}

#[derive(Op)]
#[operation(dialect = "poly", name = "eval")]
pub struct PolyEvalOp {
    #[_def]
    result: Val,  // Scalar result
    
    #[_use(type_constraint = "polynomial")]
    polynomial: Val,
    
    #[_use]
    point: Val,  // Evaluation point
}

// Register the dialect
pub fn register(registry: &mut DialectRegistry) {
    registry.register("poly", |dialect| {
        dialect.add_type::<PolynomialType>();
        // Operations auto-registered via inventory
    });
}

// Example usage:
// %0 = poly.constant [1.0, 2.0, 3.0] : !poly.polynomial<f32>
// %1 = poly.constant [4.0, 5.0] : !poly.polynomial<f32>  
// %2 = poly.add %0, %1 : !poly.polynomial<f32>
// %3 = poly.eval %2, %x : f32
```

## Built-in Dialects

uvir provides several built-in dialects that you can study as examples:

### arith - Arithmetic Operations
- Integer and floating-point arithmetic
- Bitwise operations
- Comparisons and casts
- ~30 operations total

### builtin - Core IR Constructs
- Module operation
- UnrealizedConversionCast
- Basic attributes

### scf - Structured Control Flow
- for, while, if constructs
- yield and condition operations
- Parallel and reduction operations

### func - Function Operations
- Function definitions
- Call and return
- Function constants

### affine - Affine Optimizations
- Affine loops and conditionals
- Affine maps and expressions
- Load/store with affine indexing

## Best Practices

### Naming Conventions

- **Dialect names**: Lowercase, short, memorable (e.g., "arith", "scf", "tensor")
- **Operation names**: Lowercase with underscores (e.g., "add_f", "matrix_mul")
- **Type names**: Lowercase, descriptive (e.g., "tensor", "polynomial")
- **Attribute names**: Lowercase with underscores (e.g., "memory_space", "alignment")

### Design Guidelines

1. **Minimize dialect dependencies**: Keep dialects focused and independent
2. **Use standard types**: Prefer builtin types when appropriate
3. **Provide builders**: Add convenient builder methods for complex operations
4. **Document semantics**: Clear documentation of operation behavior
5. **Test thoroughly**: Unit tests for parsing, printing, and verification

### Performance Tips

1. **Small types**: Keep dialect types under 16 bytes for inline storage
2. **Avoid allocations**: Use SmallVec and inline storage
3. **Cache type lookups**: Store TypeIds instead of recreating
4. **Minimal verification**: Do expensive checks only in debug mode

## Type Constraints

Define custom type constraints for your dialect:

```rust
pub fn register_constraints(registry: &mut ConstraintRegistry) {
    registry.add("tensor", |ctx, type_id| {
        ctx.get_dialect_type::<TensorType>(type_id).is_ok()
    });
    
    registry.add("tensor_f32", |ctx, type_id| {
        if let Ok(tensor) = ctx.get_dialect_type::<TensorType>(type_id) {
            tensor.dtype == ctx.get_f32_type()
        } else {
            false
        }
    });
}
```

## Dialect Interactions

Operations can work across dialects:

```rust
// Convert between tensor and memref
%memref = tensor.to_memref %tensor : tensor<4x4xf32> to memref<4x4xf32>

// Use arithmetic ops on custom types
%sum = arith.addf %tensor_elem1, %tensor_elem2 : f32
```

## Testing Your Dialect

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_print() {
        let input = "!poly.polynomial<f32, 10>";
        let mut ctx = Context::new();
        let mut parser = Parser::new(&mut ctx, input);
        
        let ty = PolynomialType::parse(&mut parser).unwrap();
        assert_eq!(ty.coefficient_type, ctx.get_f32_type());
        assert_eq!(ty.degree_bound, Some(10));
        
        let mut output = String::new();
        let mut printer = Printer::new(&mut output);
        ty.print(&mut printer).unwrap();
        assert_eq!(output, input);
    }
    
    #[test] 
    fn test_operation_verify() {
        let mut ctx = Context::new();
        let poly_type = PolynomialType::get(&mut ctx, ctx.get_f32_type(), None);
        
        let p1 = ctx.create_value(poly_type);
        let p2 = ctx.create_value(poly_type);
        let result = ctx.create_value(poly_type);
        
        let op = PolyAddOp { result, lhs: p1, rhs: p2 };
        assert!(op.verify(&ctx).is_ok());
    }
}
```

## Integration with Passes

Make your dialect operations work with the pass infrastructure:

```rust
impl Pattern for PolynomialFoldConstants {
    fn match_and_rewrite(
        &self,
        op: OpRef,
        rewriter: &mut PatternRewriter,
    ) -> Result<bool> {
        if let Some(add) = op.dyn_cast::<PolyAddOp>() {
            // Check if both operands are constants
            if let (Some(c1), Some(c2)) = (
                get_defining_op::<PolyConstantOp>(add.lhs),
                get_defining_op::<PolyConstantOp>(add.rhs),
            ) {
                // Fold the constants
                let new_coeffs = fold_polynomial_add(c1.coefficients, c2.coefficients);
                let new_const = PolyConstantOp {
                    result: add.result,
                    coefficients: new_coeffs,
                };
                rewriter.replace_op(op, new_const)?;
                return Ok(true);
            }
        }
        Ok(false)
    }
}
```