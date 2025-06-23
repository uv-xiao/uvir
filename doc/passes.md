# Pass Infrastructure

uvir provides a flexible pass infrastructure for transforming and optimizing IR. The system supports both pattern-based rewrites and full IR transformation passes.

## Overview

The pass system consists of two main components:

1. **Pattern Rewriting**: Local transformations based on pattern matching
2. **IR Passes**: Global transformations that analyze and modify entire modules

## Pattern-Based Rewriting

### RewritePattern Trait

Define patterns that match and transform specific operations:

```rust
pub trait RewritePattern: 'static {
    /// Higher benefit patterns are applied first
    fn benefit(&self) -> usize { 1 }
    
    /// Try to match and rewrite an operation
    fn match_and_rewrite(
        &self,
        op: OpRef,
        rewriter: &mut PatternRewriter,
        ctx: &Context,
    ) -> Result<bool>;
}
```

### Simple Pattern Example

```rust
struct FoldConstantAdd;

impl RewritePattern for FoldConstantAdd {
    fn benefit(&self) -> usize { 10 }  // High priority
    
    fn match_and_rewrite(
        &self,
        op: OpRef,
        rewriter: &mut PatternRewriter,
        ctx: &Context,
    ) -> Result<bool> {
        // Check if this is an add operation
        let Some(add) = op.dyn_cast::<arith::AddIOp>() else {
            return Ok(false);
        };
        
        // Check if both operands are constants
        let lhs_const = get_defining_op::<arith::ConstantOp>(add.lhs);
        let rhs_const = get_defining_op::<arith::ConstantOp>(add.rhs);
        
        if let (Some(lhs), Some(rhs)) = (lhs_const, rhs_const) {
            // Extract constant values
            let lhs_val = lhs.value.as_integer()?;
            let rhs_val = rhs.value.as_integer()?;
            
            // Create new constant with folded value
            let folded = arith::ConstantOp {
                result: add.result,
                value: Attribute::Integer(lhs_val + rhs_val),
            };
            
            // Replace the add with the constant
            rewriter.replace_op(op, folded.into_op_data(ctx))?;
            return Ok(true);
        }
        
        Ok(false)
    }
}
```

### PatternRewriter API

The rewriter provides methods for transforming IR:

```rust
impl PatternRewriter<'_> {
    /// Replace an operation with a new one
    pub fn replace_op(&mut self, old: OpRef, new: OpData) -> Result<()>;
    
    /// Replace an operation with multiple operations
    pub fn replace_op_with_multiple(&mut self, old: OpRef, new: Vec<OpData>) -> Result<()>;
    
    /// Erase an operation (must have no uses)
    pub fn erase_op(&mut self, op: OpRef) -> Result<()>;
    
    /// Replace all uses of a value
    pub fn replace_all_uses(&mut self, from: Val, to: Val) -> Result<()>;
    
    /// Create a new operation
    pub fn create_op(&mut self, op: OpData) -> OpRef;
    
    /// Modify operation in place
    pub fn update_op_in_place(&mut self, op: OpRef, f: impl FnOnce(&mut OpData)) -> Result<()>;
}
```

### Applying Patterns

```rust
// Create pattern set
let patterns: Vec<Box<dyn RewritePattern>> = vec![
    Box::new(FoldConstantAdd),
    Box::new(FoldConstantMul),
    Box::new(EliminateIdentity),
    Box::new(CommonSubexpressionElimination),
];

// Apply patterns greedily until fixed point
let changed = apply_patterns_greedy(
    &mut ctx,
    &patterns,
    module_region,
)?;
```

## Full IR Passes

### Pass Trait

Define passes that transform entire modules:

```rust
pub trait Pass {
    /// Unique pass name
    fn name(&self) -> &str;
    
    /// Run the pass on the context
    fn run(&mut self, ctx: &mut Context) -> Result<PassResult>;
    
    /// Dependencies that must run before this pass
    fn dependencies(&self) -> &[&str] { &[] }
    
    /// Passes that this invalidates
    fn invalidates(&self) -> &[&str] { &[] }
}

pub struct PassResult {
    pub changed: bool,
    pub statistics: HashMap<String, u64>,
}
```

### Example: Dead Code Elimination

```rust
struct DeadCodeElimination;

impl Pass for DeadCodeElimination {
    fn name(&self) -> &str { "dce" }
    
    fn run(&mut self, ctx: &mut Context) -> Result<PassResult> {
        let mut stats = HashMap::new();
        let mut changed = false;
        
        // Build use-def chains
        let use_def = build_use_def_chains(ctx);
        
        // Find dead operations (no uses of results)
        let mut worklist = Vec::new();
        for (op_ref, op_data) in ctx.walk_ops() {
            if op_data.results.iter().all(|&v| use_def.get_uses(v).is_empty()) 
                && !has_side_effects(op_data) {
                worklist.push(op_ref);
            }
        }
        
        // Remove dead operations
        while let Some(dead_op) = worklist.pop() {
            // Add operands to worklist if they become dead
            for &operand in &ctx.get_op(dead_op).operands {
                if let Some(def_op) = use_def.get_defining_op(operand) {
                    if use_def.get_uses(operand).len() == 1 {
                        worklist.push(def_op);
                    }
                }
            }
            
            // Erase the dead operation
            ctx.erase_op(dead_op)?;
            stats.insert("ops_removed".to_string(), 
                        stats.get("ops_removed").unwrap_or(&0) + 1);
            changed = true;
        }
        
        Ok(PassResult { changed, statistics: stats })
    }
}
```

### Example: Inlining Pass

```rust
struct FunctionInlining {
    max_size: usize,
}

impl Pass for FunctionInlining {
    fn name(&self) -> &str { "inline" }
    
    fn dependencies(&self) -> &[&str] { &["function-analysis"] }
    
    fn run(&mut self, ctx: &mut Context) -> Result<PassResult> {
        let mut stats = HashMap::new();
        let mut changed = false;
        
        // Collect all call sites
        let call_sites = ctx.walk_ops()
            .filter_map(|(ref_id, _)| {
                ctx.get_op(ref_id).dyn_cast::<func::CallOp>()
                    .map(|call| (ref_id, call))
            })
            .collect::<Vec<_>>();
        
        for (call_ref, call_op) in call_sites {
            // Get called function
            let Some(func) = ctx.lookup_symbol(&call_op.callee) else {
                continue;
            };
            
            // Check inlining criteria
            if should_inline(&func, self.max_size) {
                inline_function(ctx, call_ref, &func)?;
                stats.insert("functions_inlined".to_string(),
                           stats.get("functions_inlined").unwrap_or(&0) + 1);
                changed = true;
            }
        }
        
        Ok(PassResult { changed, statistics: stats })
    }
}
```

## Pass Manager

The PassManager orchestrates pass execution:

```rust
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    pipelines: HashMap<String, Vec<String>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            pipelines: HashMap::new(),
        }
    }
    
    /// Register a pass
    pub fn register_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }
    
    /// Define a named pipeline
    pub fn add_pipeline(&mut self, name: &str, passes: Vec<&str>) {
        self.pipelines.insert(name.to_string(), 
                            passes.into_iter().map(String::from).collect());
    }
    
    /// Run a specific pass
    pub fn run_pass(&mut self, ctx: &mut Context, name: &str) -> Result<PassResult> {
        let pass = self.passes.iter_mut()
            .find(|p| p.name() == name)
            .ok_or(Error::PassNotFound(name.to_string()))?;
        
        pass.run(ctx)
    }
    
    /// Run a pipeline
    pub fn run_pipeline(&mut self, ctx: &mut Context, pipeline: &str) -> Result<()> {
        let pass_names = self.pipelines.get(pipeline)
            .ok_or(Error::PipelineNotFound(pipeline.to_string()))?
            .clone();
        
        for pass_name in pass_names {
            self.run_pass(ctx, &pass_name)?;
        }
        
        Ok(())
    }
}
```

### Standard Pipelines

```rust
fn register_standard_pipelines(pm: &mut PassManager) {
    // O0 - No optimization
    pm.add_pipeline("O0", vec![
        "verify",
    ]);
    
    // O1 - Basic optimizations
    pm.add_pipeline("O1", vec![
        "verify",
        "canonicalize",
        "cse",
        "dce",
        "verify",
    ]);
    
    // O2 - Standard optimizations
    pm.add_pipeline("O2", vec![
        "verify",
        "canonicalize",
        "inline",
        "canonicalize",
        "cse",
        "licm",
        "dce",
        "verify",
    ]);
    
    // O3 - Aggressive optimizations
    pm.add_pipeline("O3", vec![
        "verify",
        "aggressive-inline",
        "canonicalize",
        "cse",
        "licm",
        "loop-unroll",
        "vectorize",
        "dce",
        "verify",
    ]);
}
```

## Analysis Infrastructure

Passes can use analysis results:

```rust
pub trait Analysis {
    type Result;
    
    fn run(&mut self, ctx: &Context) -> Self::Result;
}

// Example: Dominance analysis
struct DominanceAnalysis;

impl Analysis for DominanceAnalysis {
    type Result = DominanceInfo;
    
    fn run(&mut self, ctx: &Context) -> DominanceInfo {
        // Build dominator tree
        let mut dom_info = DominanceInfo::new();
        
        for region in ctx.regions() {
            let dom_tree = build_dominator_tree(region);
            dom_info.add_region(region.id(), dom_tree);
        }
        
        dom_info
    }
}

// Use in a pass
impl Pass for LoopInvariantCodeMotion {
    fn run(&mut self, ctx: &mut Context) -> Result<PassResult> {
        // Run prerequisite analysis
        let dom_info = DominanceAnalysis.run(ctx);
        let loop_info = LoopAnalysis.run(ctx);
        
        // Use analysis results for optimization
        for loop_data in loop_info.loops() {
            hoist_invariant_code(ctx, loop_data, &dom_info)?;
        }
        
        Ok(PassResult { changed: true, statistics: HashMap::new() })
    }
}
```

## Custom Pattern Sets

Group related patterns:

```rust
pub struct ArithmeticCanonicalization;

impl PatternSet for ArithmeticCanonicalization {
    fn patterns(&self) -> Vec<Box<dyn RewritePattern>> {
        vec![
            // Constant folding
            Box::new(FoldConstantAdd),
            Box::new(FoldConstantMul),
            Box::new(FoldConstantSub),
            
            // Identity elimination
            Box::new(AddZeroElimination),  // x + 0 -> x
            Box::new(MulOneElimination),   // x * 1 -> x
            Box::new(MulZeroSimplify),     // x * 0 -> 0
            
            // Commutative normalization
            Box::new(CommutativeNormalize), // Ensure constants on RHS
            
            // Associativity
            Box::new(ReassociateAdd),
            Box::new(ReassociateMul),
            
            // Distribution
            Box::new(DistributeMulOverAdd), // a * (b + c) -> a*b + a*c
        ]
    }
}
```

## Performance Considerations

### Pattern Matching
- Order patterns by benefit (highest first)
- Use early exits in match functions
- Cache analysis results between patterns
- Batch related transformations

### Pass Efficiency
- Minimize IR traversals
- Use incremental updates when possible
- Track changes to avoid redundant work
- Profile pass execution time

### Memory Usage
- Reuse data structures across passes
- Clear temporary analysis data
- Use streaming algorithms for large IR

## Testing Passes

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constant_folding() {
        let input = r#"
            func.func @test(%arg0: i32) -> i32 {
                %0 = arith.constant 5 : i32
                %1 = arith.constant 3 : i32
                %2 = arith.addi %0, %1 : i32
                func.return %2 : i32
            }
        "#;
        
        let expected = r#"
            func.func @test(%arg0: i32) -> i32 {
                %0 = arith.constant 8 : i32
                func.return %0 : i32
            }
        "#;
        
        let mut ctx = Context::new();
        ctx.parse_mlir(input)?;
        
        let patterns = vec![Box::new(FoldConstantAdd)];
        apply_patterns_greedy(&mut ctx, &patterns, ctx.module)?;
        
        assert_eq!(ctx.to_mlir(), expected);
    }
    
    #[test]
    fn test_dce_pass() {
        let mut ctx = create_test_context();
        let mut pass = DeadCodeElimination;
        
        let result = pass.run(&mut ctx)?;
        assert!(result.changed);
        assert_eq!(result.statistics["ops_removed"], 3);
    }
}
```

## Best Practices

1. **Write focused patterns**: Each pattern should do one thing well
2. **Test thoroughly**: Unit test each pattern and pass
3. **Document behavior**: Clear documentation of transformations
4. **Consider phase ordering**: Some optimizations enable others
5. **Profile performance**: Measure pass execution time on real code
6. **Handle edge cases**: Ensure correctness for all valid IR
7. **Preserve semantics**: Transformations must maintain program meaning