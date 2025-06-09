use crate::ops::Val;
use crate::region::RegionId;
// Re-export for the derive macro to use proper paths
use crate as uvir;
use crate::Op;

// For loop operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "for")]
pub struct ForOp {
    #[_use]
    pub lower_bound: Val,
    #[_use]
    pub upper_bound: Val,
    #[_use]
    pub step: Val,
    #[_def]
    pub results: Val,  // Can be multiple in real MLIR
    #[_region]
    pub body: RegionId,
}

// If operation with optional else
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "if")]
pub struct IfOp {
    #[_use]
    pub condition: Val,
    #[_def]
    pub results: Val,  // Can be multiple in real MLIR
    #[_region]
    pub then_region: RegionId,
    #[_region]
    pub else_region: RegionId,  // Optional in real MLIR
}

// While loop operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "while")]
pub struct WhileOp {
    #[_use]
    pub init_args: Val,  // Initial values
    #[_def]
    pub results: Val,    // Results after loop
    #[_region]
    pub before_region: RegionId,  // Condition check region
    #[_region]
    pub after_region: RegionId,   // Loop body region
}

// Yield operation - terminator for SCF regions
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "yield")]
pub struct YieldOp {
    #[_use]
    pub operands: Val,  // Values to yield (can be multiple)
}

// Condition operation - used in while loops
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "condition")]
pub struct ConditionOp {
    #[_use]
    pub condition: Val,  // Boolean condition
    #[_use]
    pub args: Val,       // Arguments to pass (can be multiple)
}

// Execute region operation - executes a region exactly once
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "execute_region")]
pub struct ExecuteRegionOp {
    #[_def]
    pub results: Val,  // Results from the region
    #[_region]
    pub body: RegionId,
}

// Parallel loop operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "parallel")]
pub struct ParallelOp {
    #[_use]
    pub lower_bounds: Val,  // Can be multiple
    #[_use]
    pub upper_bounds: Val,  // Can be multiple
    #[_use]
    pub steps: Val,         // Can be multiple
    #[_region]
    pub body: RegionId,
}

// Reduce operation - used within parallel loops
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "reduce")]
pub struct ReduceOp {
    #[_use]
    pub operand: Val,
    #[_region]
    pub reduction_operator: RegionId,
}

// Reduce return operation - terminator for reduce regions
#[derive(Op, Clone, Debug)]
#[operation(dialect = "scf", name = "reduce.return")]
pub struct ReduceReturnOp {
    #[_use]
    pub result: Val,
}