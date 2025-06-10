use crate::attribute::Attribute;
use crate::ops::Val;
use crate::region::RegionId;
// Re-export for the derive macro to use proper paths
use crate as uvir;
use crate::Op;

// Affine for loop operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "for")]
pub struct AffineForOp {
    #[_use]
    pub lower_bound: Val, // Affine map result
    #[_use]
    pub upper_bound: Val, // Affine map result
    #[_use]
    pub step: Val, // Positive constant
    #[_def]
    pub results: Val, // Loop-carried values
    #[_region]
    pub body: RegionId,
    #[_attr]
    pub lower_bound_map: Attribute, // Affine map
    #[_attr]
    pub upper_bound_map: Attribute, // Affine map
}

// Affine parallel loop operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "parallel")]
pub struct AffineParallelOp {
    #[_use]
    pub lower_bounds: Val, // Can be multiple
    #[_use]
    pub upper_bounds: Val, // Can be multiple
    #[_use]
    pub steps: Val, // Can be multiple
    #[_def]
    pub results: Val, // Reduction results
    #[_region]
    pub body: RegionId,
    #[_attr]
    pub reductions: Attribute, // Reduction operations (e.g., "add", "mul")
}

// Affine if operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "if")]
pub struct AffineIfOp {
    #[_use]
    pub operands: Val, // Operands for the condition set
    #[_def]
    pub results: Val, // Results from the if/else regions
    #[_region]
    pub then_region: RegionId,
    #[_region]
    pub else_region: RegionId, // Optional in real MLIR
    #[_attr]
    pub condition_set: Attribute, // Affine set
}

// Affine apply operation - applies an affine map
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "apply")]
pub struct AffineApplyOp {
    #[_use]
    pub operands: Val, // Input indices/symbols
    #[_def]
    pub result: Val, // Computed index
    #[_attr]
    pub map: Attribute, // Affine map to apply
}

// Affine load operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "load")]
pub struct AffineLoadOp {
    #[_use]
    pub memref: Val, // Memory reference
    #[_use]
    pub indices: Val, // Access indices (can be multiple)
    #[_def]
    pub result: Val, // Loaded value
    #[_attr]
    pub map: Attribute, // Optional affine map for indices
}

// Affine store operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "store")]
pub struct AffineStoreOp {
    #[_use]
    pub value: Val, // Value to store
    #[_use]
    pub memref: Val, // Memory reference
    #[_use]
    pub indices: Val, // Access indices (can be multiple)
    #[_attr]
    pub map: Attribute, // Optional affine map for indices
}

// Affine yield operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "yield")]
pub struct AffineYieldOp {
    #[_use]
    pub operands: Val, // Values to yield
}

// Affine min operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "min")]
pub struct AffineMinOp {
    #[_use]
    pub operands: Val, // Operands for the affine map
    #[_def]
    pub result: Val, // Minimum value
    #[_attr]
    pub map: Attribute, // Affine map
}

// Affine max operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "max")]
pub struct AffineMaxOp {
    #[_use]
    pub operands: Val, // Operands for the affine map
    #[_def]
    pub result: Val, // Maximum value
    #[_attr]
    pub map: Attribute, // Affine map
}

// Affine vector load operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "vector_load")]
pub struct AffineVectorLoadOp {
    #[_use]
    pub memref: Val, // Memory reference
    #[_use]
    pub indices: Val, // Access indices
    #[_def]
    pub result: Val, // Loaded vector
    #[_attr]
    pub map: Attribute, // Optional affine map
}

// Affine vector store operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "vector_store")]
pub struct AffineVectorStoreOp {
    #[_use]
    pub value: Val, // Vector to store
    #[_use]
    pub memref: Val, // Memory reference
    #[_use]
    pub indices: Val, // Access indices
    #[_attr]
    pub map: Attribute, // Optional affine map
}

// Affine delinearize index operation
#[derive(Op, Clone, Debug)]
#[operation(dialect = "affine", name = "delinearize_index")]
pub struct AffineDelinearizeIndexOp {
    #[_use]
    pub linear_index: Val, // Linear index to delinearize
    #[_def]
    pub multi_index: Val, // Multi-dimensional indices
    #[_attr]
    pub basis: Attribute, // Basis for delinearization
}
