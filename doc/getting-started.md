# Getting Started with uvir

This guide will walk you through using uvir to build your own compiler IR.

## Installation

Add uvir to your `Cargo.toml`:

```toml
[dependencies]
uvir = "0.1"
```

## Hello World

Let's start with a simple example that creates and manipulates IR:

```rust
use uvir::prelude::*;

fn main() -> Result<()> {
    // Create a context - this holds all IR data
    let mut ctx = Context::new();
    
    // Parse some MLIR text
    let mlir = r#"
        func.func @add(%arg0: i32, %arg1: i32) -> i32 {
            %0 = arith.addi %arg0, %arg1 : i32
            func.return %0 : i32
        }
    "#;
    
    ctx.parse_mlir(mlir)?;
    
    // Print it back out
    println!("{}", ctx.to_mlir());
    
    Ok(())
}
```

## Building IR Programmatically

Instead of parsing text, you can build IR directly:

```rust
use uvir::prelude::*;
use uvir::dialects::{arith, func};

fn build_function(ctx: &mut Context) -> Result<()> {
    // Get common types
    let i32_type = ctx.get_i32_type();
    let func_type = ctx.get_function_type(&[i32_type, i32_type], &[i32_type]);
    
    // Create function
    let func_op = func::FuncOp::builder()
        .name("multiply")
        .function_type(func_type)
        .build(ctx)?;
    
    // Get function body region
    let body = func_op.body(ctx);
    
    // Get function arguments
    let arg0 = body.argument(0)?;
    let arg1 = body.argument(1)?;
    
    // Create multiply operation
    let mul_op = arith::MulIOp::builder()
        .lhs(arg0)
        .rhs(arg1)
        .build(ctx)?;
    
    let result = mul_op.result(ctx);
    
    // Create return
    func::ReturnOp::builder()
        .operands(vec![result])
        .build_in(ctx, body)?;
    
    Ok(())
}
```

## Working with Passes

Transform your IR using the pass infrastructure:

```rust
use uvir::prelude::*;
use uvir::passes::{PassManager, ConstantFolding, DeadCodeElimination};

fn optimize_ir(ctx: &mut Context) -> Result<()> {
    let mut pass_manager = PassManager::new();
    
    // Register passes
    pass_manager.register_pass(Box::new(ConstantFolding));
    pass_manager.register_pass(Box::new(DeadCodeElimination));
    
    // Define optimization pipeline
    pass_manager.add_pipeline("optimize", vec![
        "constant-folding",
        "dce",
    ]);
    
    // Run the pipeline
    pass_manager.run_pipeline(ctx, "optimize")?;
    
    Ok(())
}
```

## Creating a Custom Dialect

Let's create a simple dialect for matrix operations:

```rust
use uvir::prelude::*;

// Define a matrix type
#[derive(Clone, PartialEq, DialectType)]
#[dialect_type(dialect = "matrix", name = "matrix")]
struct MatrixType {
    rows: i64,
    cols: i64,
    element_type: TypeId,
}

// Define matrix multiplication
#[derive(Op)]
#[operation(dialect = "matrix", name = "mul")]
struct MatrixMulOp {
    #[_def]
    result: Val,
    #[_use]
    lhs: Val,
    #[_use]
    rhs: Val,
}

// Register the dialect
pub fn register_matrix_dialect(registry: &mut DialectRegistry) {
    registry.register("matrix", |dialect| {
        dialect.add_type::<MatrixType>();
        // Operations are auto-registered
    });
}

// Use the dialect
fn use_matrix_ops(ctx: &mut Context) -> Result<()> {
    // Create matrix type
    let f32 = ctx.get_f32_type();
    let mat_type = ctx.intern_dialect_type(MatrixType {
        rows: 4,
        cols: 4,
        element_type: f32,
    });
    
    // Create values
    let a = ctx.create_value(mat_type);
    let b = ctx.create_value(mat_type);
    
    // Create matrix multiply
    let mul = MatrixMulOp {
        result: ctx.create_value(mat_type),
        lhs: a,
        rhs: b,
    };
    
    ctx.create_op(mul.into_op_data(ctx))?;
    
    Ok(())
}
```

## Pattern-Based Rewrites

Define patterns to optimize your IR:

```rust
use uvir::prelude::*;
use uvir::passes::{RewritePattern, PatternRewriter};

struct SimplifyMatrixMultiply;

impl RewritePattern for SimplifyMatrixMultiply {
    fn match_and_rewrite(
        &self,
        op: OpRef,
        rewriter: &mut PatternRewriter,
        ctx: &Context,
    ) -> Result<bool> {
        // Check if this is matrix multiply
        let Some(matmul) = op.dyn_cast::<MatrixMulOp>() else {
            return Ok(false);
        };
        
        // Check for identity matrix on right
        if is_identity_matrix(ctx, matmul.rhs) {
            // Replace matmul with its left operand
            rewriter.replace_all_uses(matmul.result, matmul.lhs)?;
            rewriter.erase_op(op)?;
            return Ok(true);
        }
        
        Ok(false)
    }
}
```

## Walking and Analyzing IR

Traverse IR to collect information:

```rust
use uvir::prelude::*;

fn count_operations(ctx: &Context) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    
    // Walk all operations in the module
    ctx.walk_ops(|op_ref, op_data| {
        let key = format!("{}.{}", op_data.info.dialect, op_data.info.name);
        *counts.entry(key).or_insert(0) += 1;
        
        // Continue walking into nested regions
        WalkResult::Advance
    });
    
    counts
}

fn find_all_constants(ctx: &Context) -> Vec<Val> {
    let mut constants = Vec::new();
    
    ctx.walk_ops(|op_ref, op_data| {
        if op_data.info.dialect == "arith" && op_data.info.name == "constant" {
            constants.extend(&op_data.results);
        }
        WalkResult::Advance
    });
    
    constants
}
```

## Error Handling

uvir uses Result types throughout:

```rust
use uvir::prelude::*;

fn safe_transformation(ctx: &mut Context) -> Result<()> {
    // Parse might fail
    let module = ctx.parse_mlir("invalid syntax")?;
    
    // Type checking might fail
    let bad_type = ctx.get_dialect_type::<MatrixType>(
        ctx.get_i32_type()  // Not a matrix type!
    )?;
    
    // Verification might fail
    ctx.verify()?;
    
    Ok(())
}

fn main() {
    if let Err(e) = safe_transformation(&mut Context::new()) {
        eprintln!("Error: {}", e);
        // Error: Parse error at line 1: expected operation
    }
}
```

## Integration Example

Here's a complete example showing how to integrate uvir into a toy language compiler:

```rust
use uvir::prelude::*;

// AST for our toy language
enum Expr {
    Number(i32),
    Add(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
    Variable(String),
}

// Lower AST to uvir
fn lower_expr(
    ctx: &mut Context,
    expr: &Expr,
    vars: &HashMap<String, Val>,
) -> Result<Val> {
    match expr {
        Expr::Number(n) => {
            // Create constant
            let op = arith::ConstantOp::builder()
                .value(Attribute::Integer(*n as i64))
                .result_type(ctx.get_i32_type())
                .build(ctx)?;
            Ok(op.result(ctx))
        }
        
        Expr::Add(lhs, rhs) => {
            let lhs_val = lower_expr(ctx, lhs, vars)?;
            let rhs_val = lower_expr(ctx, rhs, vars)?;
            
            let op = arith::AddIOp::builder()
                .lhs(lhs_val)
                .rhs(rhs_val)
                .build(ctx)?;
            Ok(op.result(ctx))
        }
        
        Expr::Multiply(lhs, rhs) => {
            let lhs_val = lower_expr(ctx, lhs, vars)?;
            let rhs_val = lower_expr(ctx, rhs, vars)?;
            
            let op = arith::MulIOp::builder()
                .lhs(lhs_val)
                .rhs(rhs_val)
                .build(ctx)?;
            Ok(op.result(ctx))
        }
        
        Expr::Variable(name) => {
            vars.get(name)
                .copied()
                .ok_or_else(|| Error::UndefinedVariable(name.clone()))
        }
    }
}

// Compile a function
fn compile_function(name: &str, params: Vec<String>, body: Expr) -> Result<String> {
    let mut ctx = Context::new();
    
    // Create function
    let i32_type = ctx.get_i32_type();
    let param_types = vec![i32_type; params.len()];
    let func_type = ctx.get_function_type(&param_types, &[i32_type]);
    
    let func = func::FuncOp::builder()
        .name(name)
        .function_type(func_type)
        .build(&mut ctx)?;
    
    // Map parameters to values
    let body_region = func.body(&ctx);
    let mut vars = HashMap::new();
    for (i, param_name) in params.iter().enumerate() {
        vars.insert(param_name.clone(), body_region.argument(i)?);
    }
    
    // Lower body expression
    ctx.set_insertion_point(body_region);
    let result = lower_expr(&mut ctx, &body, &vars)?;
    
    // Add return
    func::ReturnOp::builder()
        .operands(vec![result])
        .build(&mut ctx)?;
    
    // Optimize
    let mut pm = PassManager::new();
    pm.register_standard_passes();
    pm.run_pipeline(&mut ctx, "O2")?;
    
    // Return MLIR text
    Ok(ctx.to_mlir())
}

// Example usage
fn main() -> Result<()> {
    let ast = Expr::Add(
        Box::new(Expr::Multiply(
            Box::new(Expr::Variable("x".to_string())),
            Box::new(Expr::Number(2)),
        )),
        Box::new(Expr::Number(1)),
    );
    
    let mlir = compile_function("f", vec!["x".to_string()], ast)?;
    println!("{}", mlir);
    // Output:
    // func.func @f(%arg0: i32) -> i32 {
    //   %c2_i32 = arith.constant 2 : i32
    //   %0 = arith.muli %arg0, %c2_i32 : i32
    //   %c1_i32 = arith.constant 1 : i32
    //   %1 = arith.addi %0, %c1_i32 : i32
    //   func.return %1 : i32
    // }
    
    Ok(())
}
```

## Next Steps

- Read the [Architecture](architecture.md) guide to understand uvir's design
- Learn about the [Type System](type-system.md) for advanced type handling
- Study [Operations](operations.md) to create custom operations
- Explore [Dialects](dialects.md) to build domain-specific abstractions
- Master [Passes](passes.md) for IR transformations
- Try [egg Integration](egg-integration.md) for advanced optimizations

## Tips and Tricks

1. **Use builders**: Operation builders provide a fluent API with validation
2. **Intern early**: Create types once and reuse TypeIds
3. **Verify often**: Call `ctx.verify()` after transformations
4. **Profile**: Use cargo's built-in profiler for performance analysis
5. **Test incrementally**: Test parsing, building, and transforming separately

## Common Pitfalls

1. **Forgetting to register dialects**: Operations won't parse without registration
2. **Type mismatches**: Always verify types match operation constraints  
3. **Invalid IR**: Some transformations can create invalid IR - always verify
4. **Memory leaks**: Erased operations should have no remaining uses
5. **Infinite loops**: Pattern rewrites can loop - use `changed` flags

## Getting Help

- Check the API documentation: `cargo doc --open`
- Look at test files for examples
- File issues on GitHub for bugs or questions
- Join the community discussions